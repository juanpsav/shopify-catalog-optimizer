"""
eval.py — Evaluation harness. Measures catalog ranking via MRR.

For each product, runs the 4-tier prompt set K times via search_catalog,
computes Mean Reciprocal Rank (MRR) per tier, and saves results to:
  evals/<product_id>/<timestamp>.json

The fitment-tier MRR is the primary metric for this experiment.

Usage:
    # Evaluate a single product
    python eval.py --product-id tacoma-bed-organizer

    # Evaluate all products
    python eval.py --all

    # Evaluate with custom K (default from env/config)
    python eval.py --product-id tacoma-bed-organizer --k 3

    # Label this eval (e.g. "after-mutation-A")
    python eval.py --product-id tacoma-bed-organizer --label "mutation-A"
"""

from __future__ import annotations

import asyncio
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
from catalog_client import CatalogMCPClient, SearchResult
from prompts_gen import load_prompts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_mrr(ranks: list[int | None]) -> float:
    """Mean Reciprocal Rank from a list of ranks (None = not found)."""
    if not ranks:
        return 0.0
    rr_values = [1.0 / r if r is not None else 0.0 for r in ranks]
    return sum(rr_values) / len(rr_values)


def compute_surfaced_rate(ranks: list[int | None]) -> float:
    """Fraction of queries where the product appeared in results."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None) / len(ranks)


def compute_rank1_stability(ranks: list[int | None]) -> float:
    """Fraction of queries where the product ranked #1."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r == 1) / len(ranks)


# ─── Core eval logic ─────────────────────────────────────────────────────────

async def eval_product(
    product_id: str,
    k: int = 5,
    label: str | None = None,
    search_limit: int = 20,
) -> dict[str, Any]:
    """
    Run the full K-repetition evaluation for one product.

    Returns a result dict with per-tier MRR, surfaced rate, and top competitors.
    """
    product_dir = config.PRODUCTS_DIR / product_id
    current = json.loads((product_dir / "current.json").read_text())
    tiers = load_prompts(product_id)

    product_gid: str = current["product_id"]
    shop_domain: str = config.SHOPIFY_SHOP_DOMAIN

    catalog = CatalogMCPClient()
    now = datetime.now(timezone.utc).isoformat()

    tier_results: dict[str, dict[str, Any]] = {}
    all_competitors: dict[str, list[dict]] = {}  # query → competitor list

    for tier in config.PROMPT_TIERS:
        queries: list[str] = tiers.get(tier, [])
        if not queries:
            logger.warning("No queries for tier %r in product %s", tier, product_id)
            tier_results[tier] = {
                "mrr": 0.0,
                "surfaced_rate": 0.0,
                "rank1_stability": 0.0,
                "query_count": 0,
                "repetitions": k,
            }
            continue

        query_mrrs: list[float] = []
        query_surfaced: list[float] = []
        query_rank1: list[float] = []

        for query in queries:
            rep_ranks: list[int | None] = []
            rep_competitors: list[dict] = []

            for rep in range(k):
                result: SearchResult = await catalog.search_catalog(
                    query, limit=search_limit
                )

                # Find our product rank
                rank = result.rank_of(product_gid, shop_domain) or result.rank_of(
                    product_gid
                )
                rep_ranks.append(rank)

                # Collect top-5 competitors (products that aren't ours)
                for p in result.products[:5]:
                    is_ours = (
                        p.upid == product_gid
                        or p.upid.endswith(product_gid.split("/")[-1])
                        or (shop_domain and p.shop_domain == shop_domain)
                    )
                    if not is_ours:
                        rep_competitors.append({
                            "upid": p.upid,
                            "title": p.title,
                            "shop_domain": p.shop_domain,
                            "rank": result.products.index(p) + 1,
                        })

            query_mrrs.append(compute_mrr(rep_ranks))
            query_surfaced.append(compute_surfaced_rate(rep_ranks))
            query_rank1.append(compute_rank1_stability(rep_ranks))

            if rep_competitors:
                # Deduplicate competitors by upid
                seen: set[str] = set()
                unique: list[dict] = []
                for c in rep_competitors:
                    if c["upid"] not in seen:
                        seen.add(c["upid"])
                        unique.append(c)
                all_competitors[query] = unique[:5]

        tier_results[tier] = {
            "mrr": statistics.mean(query_mrrs) if query_mrrs else 0.0,
            "mrr_std": statistics.stdev(query_mrrs) if len(query_mrrs) > 1 else 0.0,
            "surfaced_rate": statistics.mean(query_surfaced) if query_surfaced else 0.0,
            "rank1_stability": statistics.mean(query_rank1) if query_rank1 else 0.0,
            "query_count": len(queries),
            "repetitions": k,
            "per_query_mrr": {q: m for q, m in zip(queries, query_mrrs)},
        }

    # Overall fitment-tier MRR (the headline metric)
    fitment_mrr = tier_results.get("fitment", {}).get("mrr", 0.0)

    result_doc = {
        "product_id": product_id,
        "product_gid": product_gid,
        "evaluated_at": now,
        "label": label,
        "k": k,
        "fitment_mrr": fitment_mrr,
        "tiers": tier_results,
        "top_competitors": all_competitors,
    }

    return result_doc


def save_eval_result(product_id: str, result: dict[str, Any]) -> Path:
    """Save eval result to evals/<product_id>/<timestamp>.json."""
    eval_dir = config.EVALS_DIR / product_id
    eval_dir.mkdir(exist_ok=True)
    ts = result["evaluated_at"].replace(":", "-").replace("+", "Z").split(".")[0]
    label = result.get("label") or "eval"
    filename = f"{ts}-{label}.json"
    path = eval_dir / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return path


def load_latest_eval(product_id: str) -> dict[str, Any] | None:
    """Load the most recent eval result for a product."""
    eval_dir = config.EVALS_DIR / product_id
    if not eval_dir.is_dir():
        return None
    files = sorted(eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text())


def load_all_evals(product_id: str) -> list[dict[str, Any]]:
    """Load all eval results for a product, sorted oldest-first."""
    eval_dir = config.EVALS_DIR / product_id
    if not eval_dir.is_dir():
        return []
    files = sorted(eval_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    results = []
    for f in files:
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return results


def list_product_ids() -> list[str]:
    return [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", default=None, help="Single product to evaluate")
@click.option("--all", "all_products", is_flag=True, help="Evaluate all products")
@click.option("--k", default=None, type=int, help="Repetitions per query (default: EVAL_K env)")
@click.option("--label", default=None, help="Label for this eval run (e.g. 'mutation-A')")
@click.option("--no-save", is_flag=True, help="Print results but don't write to evals/")
def main(
    product_id: str | None,
    all_products: bool,
    k: int | None,
    label: str | None,
    no_save: bool,
) -> None:
    """Run catalog ranking evaluation. Reports fitment-tier MRR per product."""

    if not product_id and not all_products:
        click.echo("Pass --product-id <id> or --all. Run --help for usage.")
        sys.exit(1)

    k_val = k or config.EVAL_K
    ids = list_product_ids() if all_products else [product_id]

    async def run_all():
        for pid in ids:
            product_dir = config.PRODUCTS_DIR / pid
            if not product_dir.is_dir():
                click.echo(f"No products/{pid}/ directory — skipping", err=True)
                continue
            if not (config.PROMPTS_DIR / f"{pid}.yaml").exists():
                click.echo(f"No prompts/{pid}.yaml — run prompts_gen.py first", err=True)
                continue

            click.echo(f"\nEvaluating {pid} (K={k_val})…")
            result = await eval_product(pid, k=k_val, label=label)

            # Pretty-print summary
            click.echo(f"  fitment MRR: {result['fitment_mrr']:.4f}")
            for tier, stats in result["tiers"].items():
                mrr = stats.get("mrr", 0.0)
                surfaced = stats.get("surfaced_rate", 0.0)
                click.echo(
                    f"  {tier:12s} MRR={mrr:.4f}  surfaced={surfaced:.1%}"
                )

            if not no_save:
                path = save_eval_result(pid, result)
                click.echo(f"  Saved: {path}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
