"""
decide.py — Decision engine: KEEP, REVERT, or INCONCLUSIVE.

Compares a new eval score against the rolling baseline (± 2σ):
  new_score > baseline_mean + DECISION_SIGMA * baseline_std  → KEEP
  new_score < baseline_mean - DECISION_SIGMA * baseline_std  → REVERT (restore field)
  else                                                        → INCONCLUSIVE

After deciding:
  - Updates products/<id>/history.jsonl (status + decision fields)
  - On REVERT: calls Admin API to restore the prior field value
  - Clears products/<id>/pending_index.json
  - Optionally updates products/<id>/baseline.json with new stable data

Usage:
    # Decide based on the most recent eval for a product
    python decide.py --product-id tacoma-bed-organizer

    # Decide using a specific eval file
    python decide.py --product-id tacoma-bed-organizer \\
                     --eval-file evals/tacoma-bed-organizer/2025-01-15T12-00-00-mutation-A.json

    # Dry-run: print decision without writing or calling Admin API
    python decide.py --product-id tacoma-bed-organizer --dry-run
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

import config
from admin_client import get_admin_client
from eval import load_latest_eval, load_all_evals
from mutate import load_history, update_history_entry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Baseline loading ─────────────────────────────────────────────────────────

def load_baseline(product_id: str) -> dict[str, Any] | None:
    path = config.PRODUCTS_DIR / product_id / "baseline.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_baseline_from_evals(product_id: str) -> dict[str, Any] | None:
    """
    Build a rolling baseline from up to the last 3 stable (KEEP/INCONCLUSIVE) evals.
    Returns None if fewer than 1 eval exists (can't establish baseline).
    """
    all_evals = load_all_evals(product_id)
    history = load_history(product_id)

    # Map cycle → status from history
    cycle_status: dict[int, str] = {}
    for entry in history:
        c = entry.get("cycle")
        s = entry.get("status", "")
        if c and s:
            cycle_status[c] = s

    # Filter to stable evals: baseline label, or KEEP/INCONCLUSIVE cycles
    stable_evals = []
    for ev in all_evals:
        label = ev.get("label", "")
        if label == "baseline" or label.startswith("baseline"):
            stable_evals.append(ev)

    # Use last 3 stable evals (or all if fewer)
    stable_evals = stable_evals[-3:]

    if not stable_evals:
        # Fall back to all evals labeled with the product baseline
        return None

    # Compute per-tier mean + std
    tier_values: dict[str, list[float]] = {t: [] for t in config.PROMPT_TIERS}
    for ev in stable_evals:
        for tier in config.PROMPT_TIERS:
            mrr = ev.get("tiers", {}).get(tier, {}).get("mrr", 0.0)
            tier_values[tier].append(mrr)

    baseline: dict[str, Any] = {
        "product_id": product_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_eval_count": len(stable_evals),
        "tiers": {},
    }
    for tier, values in tier_values.items():
        if values:
            baseline["tiers"][tier] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }
        else:
            baseline["tiers"][tier] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}

    # Also store top-level fitment mean for quick access
    baseline["fitment_mrr_mean"] = baseline["tiers"]["fitment"]["mean"]
    baseline["fitment_mrr_std"] = baseline["tiers"]["fitment"]["std"]

    return baseline


def save_baseline(product_id: str, baseline: dict[str, Any]) -> Path:
    path = config.PRODUCTS_DIR / product_id / "baseline.json"
    path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
    return path


# ─── Decision logic ───────────────────────────────────────────────────────────

DECISION_KEEP = "KEEP"
DECISION_REVERT = "REVERT"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"


def decide(
    new_fitment_mrr: float,
    baseline_mean: float,
    baseline_std: float,
    sigma: float = None,
) -> str:
    """
    Apply the decision rule based on fitment-tier MRR.

    Returns KEEP, REVERT, or INCONCLUSIVE.
    """
    sigma = sigma or config.DECISION_SIGMA

    # Handle zero-std baseline (all zeros → any positive lift is KEEP)
    effective_std = baseline_std if baseline_std > 1e-9 else 0.001

    upper_threshold = baseline_mean + sigma * effective_std
    lower_threshold = baseline_mean - sigma * effective_std

    if new_fitment_mrr > upper_threshold:
        return DECISION_KEEP
    if new_fitment_mrr < lower_threshold:
        return DECISION_REVERT
    return DECISION_INCONCLUSIVE


def make_decision(
    product_id: str,
    eval_result: dict[str, Any],
    dry_run: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Full decision cycle:
    1. Load or build baseline
    2. Compare new eval vs baseline
    3. Execute KEEP / REVERT / INCONCLUSIVE
    4. Update history

    Returns (decision_str, decision_detail_dict).
    """
    # Load baseline
    baseline = load_baseline(product_id)
    if baseline is None:
        baseline = build_baseline_from_evals(product_id)
    if baseline is None:
        # No baseline yet — use zero baseline (everything is "lift")
        baseline = {
            "tiers": {t: {"mean": 0.0, "std": 0.0} for t in config.PROMPT_TIERS},
            "fitment_mrr_mean": 0.0,
            "fitment_mrr_std": 0.0,
        }

    baseline_fitment_mean = baseline.get("fitment_mrr_mean", 0.0)
    baseline_fitment_std = baseline.get("fitment_mrr_std", 0.0)
    new_fitment_mrr = eval_result.get("fitment_mrr", 0.0)

    decision = decide(new_fitment_mrr, baseline_fitment_mean, baseline_fitment_std)

    # Find the pending history entry to update
    history = load_history(product_id)
    pending_entry = next(
        (e for e in reversed(history) if e.get("status") == "PENDING"),
        None,
    )

    detail = {
        "product_id": product_id,
        "decision": decision,
        "new_fitment_mrr": new_fitment_mrr,
        "baseline_fitment_mean": baseline_fitment_mean,
        "baseline_fitment_std": baseline_fitment_std,
        "threshold_keep": baseline_fitment_mean + config.DECISION_SIGMA * max(baseline_fitment_std, 0.001),
        "threshold_revert": baseline_fitment_mean - config.DECISION_SIGMA * max(baseline_fitment_std, 0.001),
        "delta": new_fitment_mrr - baseline_fitment_mean,
        "cycle": pending_entry.get("cycle") if pending_entry else None,
        "mutation_class": pending_entry.get("mutation_class") if pending_entry else None,
        "field": pending_entry.get("field") if pending_entry else None,
    }

    if decision == DECISION_REVERT and pending_entry and not dry_run:
        prior_value = pending_entry.get("prior_value")
        field = pending_entry.get("field")
        product_gid = eval_result.get("product_gid", "")
        if prior_value is not None and field and product_gid:
            admin = get_admin_client()
            admin.restore_field(product_gid, field, prior_value)
            logger.info("REVERT: restored %s.%s", product_id, field)
        else:
            logger.warning("REVERT requested but prior_value or field missing — cannot restore")

    # Update history entry
    if pending_entry and not dry_run:
        cycle = pending_entry.get("cycle")
        update_history_entry(product_id, cycle, {
            "status": decision,
            "eval_score": new_fitment_mrr,
            "baseline_score": baseline_fitment_mean,
            "decision": decision,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        })

    # Clear pending_index.json after decision
    pending_path = config.PRODUCTS_DIR / product_id / "pending_index.json"
    if pending_path.exists() and not dry_run:
        pending_path.unlink()

    # If KEEP, update current.json with the new field value
    if decision == DECISION_KEEP and pending_entry and not dry_run:
        _update_current_json(product_id, pending_entry)

    return decision, detail


def _update_current_json(product_id: str, history_entry: dict[str, Any]) -> None:
    """Apply the kept mutation to current.json."""
    current_path = config.PRODUCTS_DIR / product_id / "current.json"
    current = json.loads(current_path.read_text())

    field = history_entry.get("field", "")
    new_value = history_entry.get("new_value")

    # Map API field names to current.json keys
    field_map = {
        "title": "title",
        "descriptionHtml": "description_html",
        "tags": "tags",
        "productType": "product_type",
        "metafields": "metafields",
    }
    json_key = field_map.get(field, field)
    if json_key in current and new_value is not None:
        current[json_key] = new_value
        current_path.write_text(json.dumps(current, indent=2, ensure_ascii=False))


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", required=True, help="Local product ID")
@click.option("--eval-file", default=None, type=click.Path(), help="Specific eval JSON file")
@click.option("--dry-run", is_flag=True, help="Print decision without writing or calling Admin API")
def main(
    product_id: str,
    eval_file: str | None,
    dry_run: bool,
) -> None:
    """Run the decision engine for a product's most recent eval."""

    product_dir = config.PRODUCTS_DIR / product_id
    if not product_dir.is_dir():
        click.echo(f"No products/{product_id}/ directory found.", err=True)
        sys.exit(1)

    if eval_file:
        eval_result = json.loads(Path(eval_file).read_text())
    else:
        eval_result = load_latest_eval(product_id)

    if eval_result is None:
        click.echo(f"No eval results found for {product_id}. Run eval.py first.", err=True)
        sys.exit(1)

    click.echo(f"\nDecision engine for {product_id}")
    click.echo(f"  New fitment MRR: {eval_result.get('fitment_mrr', 0.0):.4f}")

    decision, detail = make_decision(product_id, eval_result, dry_run=dry_run)

    click.echo(f"\n  ─────────────────────────────────")
    click.echo(f"  DECISION: {decision}")
    click.echo(f"  ─────────────────────────────────")
    click.echo(f"  Baseline mean: {detail['baseline_fitment_mean']:.4f}")
    click.echo(f"  Baseline std:  {detail['baseline_fitment_std']:.4f}")
    click.echo(f"  Keep threshold:   > {detail['threshold_keep']:.4f}")
    click.echo(f"  Revert threshold: < {detail['threshold_revert']:.4f}")
    click.echo(f"  Delta vs baseline: {detail['delta']:+.4f}")

    if dry_run:
        click.echo("\n  [DRY-RUN] No changes written.")


if __name__ == "__main__":
    main()
