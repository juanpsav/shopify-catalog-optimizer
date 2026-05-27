"""
run_baseline.py — Baseline evaluation runner.

Runs the full evaluation (K=EVAL_K reps) on ALL products BEFORE any mutations.
This is the "before" snapshot — the most important step in the experiment.

Outputs:
  products/<id>/baseline.json    — per-tier MRR mean ± std for each product
  evals/<id>/baseline-<ts>.json  — raw eval results (also written by eval.py)
  evals/baseline-report.md       — human-readable baseline report

Run this ONCE before any mutations.

Usage:
    python run_baseline.py

    # Force re-run even if baseline.json already exists
    python run_baseline.py --force

    # Use a smaller K for faster testing
    python run_baseline.py --k 2
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

import config
from eval import eval_product, save_eval_result
from decide import save_baseline, build_baseline_from_evals

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Baseline computation ─────────────────────────────────────────────────────

async def run_baseline_for_product(
    product_id: str,
    k: int,
    force: bool = False,
) -> dict[str, Any]:
    """
    Run baseline evaluation for one product.
    Returns the eval result dict.
    """
    baseline_path = config.PRODUCTS_DIR / product_id / "baseline.json"
    if not force and baseline_path.exists():
        existing = json.loads(baseline_path.read_text())
        click.echo(f"  {product_id}: baseline.json already exists (use --force to re-run)")
        return existing

    click.echo(f"  {product_id}: running baseline eval (K={k})…")
    result = await eval_product(product_id, k=k, label="baseline")
    save_eval_result(product_id, result)

    # Build and save baseline from this single eval
    # (We seed with one eval; subsequent stable evals will extend it to 3)
    baseline = {
        "product_id": product_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_eval_count": 1,
        "tiers": {},
    }
    for tier in config.PROMPT_TIERS:
        mrr = result.get("tiers", {}).get(tier, {}).get("mrr", 0.0)
        baseline["tiers"][tier] = {
            "mean": mrr,
            "std": 0.0,
            "min": mrr,
            "max": mrr,
            "n": 1,
        }
    baseline["fitment_mrr_mean"] = baseline["tiers"]["fitment"]["mean"]
    baseline["fitment_mrr_std"] = 0.0

    save_baseline(product_id, baseline)
    click.echo(
        f"  {product_id}: fitment MRR = {baseline['fitment_mrr_mean']:.4f}  "
        f"(head={baseline['tiers']['head']['mean']:.4f}, "
        f"mid={baseline['tiers']['mid']['mean']:.4f})"
    )

    return result


# ─── Markdown report ─────────────────────────────────────────────────────────

def generate_baseline_report(
    results: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> str:
    """Generate evals/baseline-report.md from all product baselines."""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Catalog Optimizer — Baseline Report",
        "",
        f"Generated: {now}",
        "",
        "This report captures catalog ranking **before any mutations**.",
        "The fitment-tier MRR is the primary metric; it is often 0% at baseline.",
        "",
        "---",
        "",
        "## Summary: Fitment-tier MRR by Product",
        "",
        "| Product | Head MRR | Mid MRR | Long-tail MRR | **Fitment MRR** | Surfaced Rate |",
        "| ------- | -------- | ------- | ------------- | --------------- | ------------- |",
    ]

    for pid in sorted(results.keys()):
        result = results[pid]
        tiers = result.get("tiers", {})
        head_mrr = tiers.get("head", {}).get("mrr", 0.0)
        mid_mrr = tiers.get("mid", {}).get("mrr", 0.0)
        lt_mrr = tiers.get("long_tail", {}).get("mrr", 0.0)
        fit_mrr = tiers.get("fitment", {}).get("mrr", 0.0)
        fit_surf = tiers.get("fitment", {}).get("surfaced_rate", 0.0)
        lines.append(
            f"| {pid} | {head_mrr:.3f} | {mid_mrr:.3f} | {lt_mrr:.3f} | **{fit_mrr:.3f}** | {fit_surf:.0%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Product Detail",
        "",
    ]

    for pid in sorted(results.keys()):
        result = results[pid]
        tiers = result.get("tiers", {})
        competitors = result.get("top_competitors", {})

        lines += [
            f"### {pid}",
            "",
        ]

        for tier in config.PROMPT_TIERS:
            stats = tiers.get(tier, {})
            mrr = stats.get("mrr", 0.0)
            surfaced = stats.get("surfaced_rate", 0.0)
            rank1 = stats.get("rank1_stability", 0.0)
            q_count = stats.get("query_count", 0)
            lines += [
                f"**{tier.replace('_', '-')} tier** ({q_count} queries)",
                f"- MRR: {mrr:.4f}",
                f"- Surfaced rate: {surfaced:.1%}",
                f"- Rank-1 stability: {rank1:.1%}",
                "",
            ]

            # Per-query breakdown
            per_query = stats.get("per_query_mrr", {})
            if per_query:
                lines.append("| Query | MRR |")
                lines.append("| ----- | --- |")
                for q, m in per_query.items():
                    lines.append(f"| {q} | {m:.4f} |")
                lines.append("")

        # Top competitors
        if competitors:
            lines += [
                "**Top competitors at baseline:**",
                "",
            ]
            seen_upids: set[str] = set()
            for query, comps in list(competitors.items())[:3]:
                lines.append(f"Query: *{query}*")
                for c in comps[:5]:
                    if c["upid"] not in seen_upids:
                        seen_upids.add(c["upid"])
                        lines.append(
                            f"  - Rank {c['rank']}: [{c['title']}] @ {c['shop_domain']}"
                        )
                lines.append("")

        lines += ["---", ""]

    lines += [
        "## What Happens Next",
        "",
        "1. Review this baseline. Fitment-tier MRR near 0 is expected — that's the problem this experiment fixes.",
        "2. Run `python mutate.py --product-id <id>` to apply the first mutation (Class A: title fitment).",
        "3. Wait 24–48h for re-indexing.",
        "4. `watch_index.py` (cron every 2h) will detect the re-index and auto-run eval + decide.",
        "5. After all 5 products complete Class A, compare fitment-tier MRR to this baseline.",
        "",
        "**If Class A alone produces >5× fitment MRR lift, that's the headline finding: title is everything.**",
        "",
    ]

    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--k", default=None, type=int, help="Repetitions per query (default: EVAL_K env)")
@click.option("--force", is_flag=True, help="Re-run even if baseline.json already exists")
@click.option("--product-id", default=None, help="Baseline only this one product")
def main(k: int | None, force: bool, product_id: str | None) -> None:
    """Run baseline evaluation for all products and generate baseline-report.md."""

    k_val = k or config.EVAL_K

    if product_id:
        ids = [product_id]
    else:
        ids = [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]

    if not ids:
        click.echo("No product directories found in products/. Run scrape.py first.", err=True)
        sys.exit(1)

    # Check prompt files
    missing_prompts = [pid for pid in ids if not (config.PROMPTS_DIR / f"{pid}.yaml").exists()]
    if missing_prompts:
        click.echo(
            f"Missing prompt files for: {', '.join(missing_prompts)}\n"
            "Run `python prompts_gen.py --all` first.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Running baseline for {len(ids)} product(s) (K={k_val})…\n")

    async def run_all():
        results: dict[str, dict] = {}
        baselines: dict[str, dict] = {}
        for pid in ids:
            result = await run_baseline_for_product(pid, k=k_val, force=force)
            results[pid] = result
            bl_path = config.PRODUCTS_DIR / pid / "baseline.json"
            if bl_path.exists():
                baselines[pid] = json.loads(bl_path.read_text())
        return results, baselines

    results, baselines = asyncio.run(run_all())

    # Generate markdown report
    report = generate_baseline_report(results, baselines)
    report_path = config.EVALS_DIR / "baseline-report.md"
    report_path.write_text(report, encoding="utf-8")

    click.echo(f"\n✓ Baseline report written: {report_path}")
    click.echo("\nNext step: python mutate.py --product-id <id>  (apply Class A: title fitment)")


if __name__ == "__main__":
    main()
