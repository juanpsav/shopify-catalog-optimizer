"""
watch_index.py — Index re-detection cron (runs every 2h via system cron).

For each product with a pending_index.json, checks whether the expected
field value has appeared in the Catalog MCP (via get_product / lookup_catalog).
If re-indexed:
  1. Logs an index-events.jsonl entry
  2. Triggers eval.py for that product
  3. Triggers decide.py with the new eval result
  4. Clears pending_index.json (done by decide.py on completion)

Usage:
    # Run manually
    python watch_index.py

    # Check only one product
    python watch_index.py --product-id tacoma-bed-organizer

    # Check only — don't trigger eval/decide
    python watch_index.py --check-only

Cron setup (every 2h):
    0 */2 * * * cd /path/to/catalog-optimizer && python watch_index.py >> logs/watch.log 2>&1
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

import config
from catalog_client import CatalogMCPClient, CatalogProduct

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ─── State helpers ────────────────────────────────────────────────────────────

def load_index_status() -> dict[str, Any]:
    path = config.STATE_DIR / "index-status.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"last_checked": None, "products_pending": [], "products_indexed": [], "products_evaluating": []}


def save_index_status(status: dict[str, Any]) -> None:
    path = config.STATE_DIR / "index-status.json"
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False))


def append_index_event(event: dict[str, Any]) -> None:
    path = config.STATE_DIR / "index-events.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def list_pending_products() -> list[str]:
    """Return product IDs that have a pending_index.json."""
    pending = []
    for d in config.PRODUCTS_DIR.iterdir():
        if d.is_dir() and (d / "pending_index.json").exists():
            pending.append(d.name)
    return sorted(pending)


# ─── Re-index detection ───────────────────────────────────────────────────────

def _values_match(expected: Any, actual: Any) -> bool:
    """
    Loose comparison between expected and indexed values.
    Handles title (string prefix/contains), tags (set inclusion), HTML (text extraction).
    """
    if expected is None:
        return True

    if isinstance(expected, str) and isinstance(actual, str):
        # For title: check that the expected value appears (case-insensitive)
        return expected.lower() in actual.lower() or actual.lower() in expected.lower()

    if isinstance(expected, list) and isinstance(actual, (list, str)):
        if isinstance(actual, str):
            actual = [actual]
        exp_set = {str(e).lower() for e in expected}
        act_set = {str(a).lower() for a in actual}
        return bool(exp_set & act_set)  # At least one expected tag is present

    return str(expected)[:50] == str(actual)[:50]


async def check_product_indexed(
    product_id: str,
    pending: dict[str, Any],
    catalog: CatalogMCPClient,
) -> bool:
    """
    Check whether the expected value from pending_index.json is now live
    in the Catalog for this product.

    Strategy:
    1. Try lookup_catalog with shop domain + product GID
    2. Fall back to searching by expected field value
    """
    product_dir = config.PRODUCTS_DIR / product_id
    current = json.loads((product_dir / "current.json").read_text())

    product_gid: str = current["product_id"]
    shop_domain: str = config.SHOPIFY_SHOP_DOMAIN
    expected_field = pending.get("field", "")
    expected_value = pending.get("expected_value")

    # Attempt 1: Direct lookup
    catalog_product: CatalogProduct | None = None
    if shop_domain:
        catalog_product = await catalog.lookup_catalog(shop_domain, product_gid)

    if catalog_product is None:
        logger.debug("lookup_catalog returned None for %s", product_id)
        return False

    # Check indexed field value against expected
    field_map = {
        "title": "title",
        "descriptionHtml": "description",
        "tags": "tags",
        "productType": "product_type",
    }
    catalog_field = field_map.get(expected_field, expected_field)
    actual_value = getattr(catalog_product, catalog_field, None) or catalog_product.raw.get(catalog_field)

    if actual_value is None:
        logger.debug("Field %r not found in catalog response for %s", catalog_field, product_id)
        # Can't confirm — treat as not yet indexed
        return False

    matched = _values_match(expected_value, actual_value)
    logger.debug(
        "Index check %s: expected[%s]=%r, actual=%r, match=%s",
        product_id, expected_field, str(expected_value)[:80], str(actual_value)[:80], matched,
    )
    return matched


def is_past_min_check_time(pending: dict[str, Any]) -> bool:
    """Return True if the minimum re-check window has passed."""
    min_check = pending.get("min_check_after")
    if not min_check:
        return True
    try:
        min_dt = datetime.fromisoformat(min_check)
        return datetime.now(timezone.utc) >= min_dt
    except (ValueError, TypeError):
        return True


# ─── Trigger eval + decide ────────────────────────────────────────────────────

async def trigger_eval_and_decide(product_id: str) -> None:
    """Run eval.py then decide.py for a product in-process."""
    import importlib

    # Import and call eval
    from eval import eval_product, save_eval_result
    from decide import make_decision

    click.echo(f"  [{product_id}] Running evaluation…")
    try:
        from mutate import load_history
        history = load_history(product_id)
        pending_entry = next((e for e in reversed(history) if e.get("status") == "PENDING"), None)
        cycle = pending_entry.get("cycle") if pending_entry else 0
        label = f"mutation-{pending_entry.get('mutation_class', 'X')}-cycle{cycle}" if pending_entry else "post-reindex"

        result = await eval_product(product_id, k=config.EVAL_K, label=label)
        eval_path = save_eval_result(product_id, result)
        click.echo(f"  [{product_id}] Eval complete — fitment MRR: {result['fitment_mrr']:.4f}")
        click.echo(f"  [{product_id}] Saved to: {eval_path}")

        decision, detail = make_decision(product_id, result)
        click.echo(f"  [{product_id}] DECISION: {decision}  (delta: {detail['delta']:+.4f})")

        append_index_event({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "product_id": product_id,
            "event": "reindex_detected_and_evaluated",
            "decision": decision,
            "fitment_mrr": result["fitment_mrr"],
            "delta": detail["delta"],
            "cycle": detail.get("cycle"),
            "mutation_class": detail.get("mutation_class"),
        })

    except Exception as exc:
        logger.error("eval/decide failed for %s: %s", product_id, exc, exc_info=True)


# ─── Main watch loop ──────────────────────────────────────────────────────────

async def watch(
    product_ids: list[str] | None = None,
    check_only: bool = False,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    catalog = CatalogMCPClient()
    status = load_index_status()
    status["last_checked"] = now

    targets = product_ids or list_pending_products()
    if not targets:
        logger.info("No products pending re-index. Nothing to do.")
        save_index_status(status)
        return

    logger.info("Checking %d product(s) for re-index: %s", len(targets), targets)

    newly_indexed: list[str] = []

    for pid in targets:
        pending_path = config.PRODUCTS_DIR / pid / "pending_index.json"
        if not pending_path.exists():
            logger.debug("No pending_index.json for %s — skipping", pid)
            continue

        pending = json.loads(pending_path.read_text())

        if pending.get("dry_run"):
            logger.info("[%s] Dry-run mutation — skipping index check, triggering eval directly", pid)
            if not check_only:
                await trigger_eval_and_decide(pid)
            continue

        if not is_past_min_check_time(pending):
            min_check = pending.get("min_check_after", "?")
            logger.info("[%s] Too early — min_check_after: %s", pid, min_check)
            continue

        logger.info("[%s] Checking index…", pid)
        indexed = await check_product_indexed(pid, pending, catalog)

        if indexed:
            logger.info("[%s] ✓ Re-index detected!", pid)
            newly_indexed.append(pid)
            append_index_event({
                "timestamp": now,
                "product_id": pid,
                "event": "reindex_detected",
                "field": pending.get("field"),
                "cycle": pending.get("mutation_cycle"),
            })
            if not check_only:
                await trigger_eval_and_decide(pid)
        else:
            logger.info("[%s] Not yet re-indexed (field not updated in catalog)", pid)

    status["products_pending"] = [
        p for p in status.get("products_pending", []) if p not in newly_indexed
    ]
    status["products_indexed"] = (status.get("products_indexed", []) + newly_indexed)[-50:]
    save_index_status(status)

    click.echo(f"\nSummary: {len(newly_indexed)}/{len(targets)} product(s) re-indexed this run")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", default=None, help="Check only this product")
@click.option("--check-only", is_flag=True, help="Detect re-index but don't trigger eval/decide")
def main(product_id: str | None, check_only: bool) -> None:
    """Detect catalog re-indexes and trigger eval + decide pipeline."""
    ids = [product_id] if product_id else None
    asyncio.run(watch(product_ids=ids, check_only=check_only))


if __name__ == "__main__":
    main()
