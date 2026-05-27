"""
pipeline.py — Main orchestrator. Runs one full cycle per product.

A "cycle" is:
  mutate → record pending → (24–48h wait for re-index) → eval → decide

This script handles the pre-reindex half: mutate + write pending.
The post-reindex half (eval + decide) is triggered by watch_index.py.

Usage:
    # Run one cycle for all products that don't have a pending mutation
    python pipeline.py run

    # Run one cycle for a specific product
    python pipeline.py run --product-id tacoma-bed-organizer

    # Run baseline first, then start mutations
    python pipeline.py start

    # Show status of all products
    python pipeline.py status

    # Run everything in dry-run mode (no API calls)
    python pipeline.py run --dry-run
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

import config
from mutate import load_history, next_mutation_class, current_cycle_number

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Status helpers ───────────────────────────────────────────────────────────

def product_status(product_id: str) -> dict[str, Any]:
    """Return a status summary for one product."""
    product_dir = config.PRODUCTS_DIR / product_id
    if not product_dir.is_dir():
        return {"product_id": product_id, "status": "missing"}

    current_exists = (product_dir / "current.json").exists()
    fitment_exists = (product_dir / "fitment.json").exists()
    prompts_exist = (config.PROMPTS_DIR / f"{product_id}.yaml").exists()
    baseline_exists = (product_dir / "baseline.json").exists()
    pending_path = product_dir / "pending_index.json"
    has_pending = pending_path.exists()

    history = load_history(product_id)
    next_cls = next_mutation_class(history)
    cycle = current_cycle_number(history)

    # Eval summary
    eval_dir = config.EVALS_DIR / product_id
    eval_count = len(list(eval_dir.glob("*.json"))) if eval_dir.exists() else 0

    status: dict[str, Any] = {
        "product_id": product_id,
        "ready": current_exists and fitment_exists,
        "prompts_generated": prompts_exist,
        "baseline_done": baseline_exists,
        "has_pending_mutation": has_pending,
        "current_cycle": cycle,
        "next_mutation_class": next_cls,
        "eval_count": eval_count,
        "completed_classes": [],
        "decisions": {},
    }

    for entry in history:
        cls = entry.get("mutation_class", "")
        decision = entry.get("decision") or entry.get("status", "")
        if cls and decision and decision != "PENDING":
            status["completed_classes"].append(cls)
            status["decisions"][cls] = decision

    if has_pending:
        pending = json.loads(pending_path.read_text())
        status["pending"] = {
            "class": pending.get("mutation_class"),
            "field": pending.get("field"),
            "written_at": pending.get("written_at"),
            "min_check_after": pending.get("min_check_after"),
        }

    return status


def print_status_table(statuses: list[dict[str, Any]]) -> None:
    """Print a rich status table."""
    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(box=box.ROUNDED, title="Catalog Optimizer — Product Status")
        table.add_column("Product", style="cyan")
        table.add_column("Ready", justify="center")
        table.add_column("Baseline", justify="center")
        table.add_column("Cycle", justify="right")
        table.add_column("Pending", justify="center")
        table.add_column("Next Class", justify="center")
        table.add_column("Decisions")

        for s in statuses:
            decisions_str = " ".join(
                f"{c}:{d[0]}" for c, d in sorted(s.get("decisions", {}).items())
            ) or "—"
            pending_str = s["pending"]["class"] if s.get("has_pending_mutation") else "—"
            next_cls = s.get("next_mutation_class") or "done"

            table.add_row(
                s["product_id"],
                "✓" if s.get("ready") else "✗",
                "✓" if s.get("baseline_done") else "✗",
                str(s.get("current_cycle", 1)),
                pending_str,
                next_cls,
                decisions_str,
            )

        console.print(table)
    except ImportError:
        # Fallback plain text
        for s in statuses:
            click.echo(
                f"{s['product_id']:35s} "
                f"ready={s.get('ready', False)!s:5} "
                f"baseline={s.get('baseline_done', False)!s:5} "
                f"cycle={s.get('current_cycle', 1)} "
                f"next={s.get('next_mutation_class') or 'done'} "
                f"decisions={s.get('decisions', {})}"
            )


# ─── Pipeline commands ────────────────────────────────────────────────────────

def run_script(script: str, args: list[str]) -> int:
    """Run a pipeline script as a subprocess and return exit code."""
    cmd = [sys.executable, script] + args
    click.echo(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(config.ROOT))
    return result.returncode


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Catalog Optimizer pipeline orchestrator."""


@cli.command()
@click.option("--product-id", default=None, help="Limit to one product")
def status(product_id: str | None) -> None:
    """Show status for all products."""
    ids = (
        [product_id]
        if product_id
        else [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]
    )
    statuses = [product_status(pid) for pid in ids]
    print_status_table(statuses)


@cli.command()
@click.option("--product-id", default=None, help="Limit to one product")
@click.option("--dry-run", is_flag=True)
@click.option("--model", default=None)
def run(product_id: str | None, dry_run: bool, model: str | None) -> None:
    """
    Run one mutation cycle for all products that don't have a pending mutation.
    After this command, wait 24-48h for re-indexing, then watch_index.py
    will auto-trigger eval + decide.
    """
    ids = (
        [product_id]
        if product_id
        else [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]
    )

    for pid in ids:
        s = product_status(pid)

        if not s.get("ready"):
            click.echo(f"  {pid}: not ready — run scrape.py first")
            continue

        if not s.get("prompts_generated"):
            click.echo(f"  {pid}: no prompts — run prompts_gen.py first")
            continue

        if not s.get("baseline_done"):
            click.echo(f"  {pid}: no baseline — run run_baseline.py first")
            continue

        if s.get("has_pending_mutation"):
            pending = s.get("pending", {})
            click.echo(
                f"  {pid}: pending class-{pending.get('class')} mutation "
                f"(min check: {pending.get('min_check_after', '?')}) — skipping"
            )
            continue

        next_cls = s.get("next_mutation_class")
        if next_cls is None:
            click.echo(f"  {pid}: all mutation classes exhausted")
            continue

        click.echo(f"\n→ {pid}: applying class {next_cls}")
        args = ["--product-id", pid]
        if dry_run:
            args.append("--dry-run")
        if model:
            args += ["--model", model]
        run_script("mutate.py", args)

    click.echo(
        "\nMutations queued. Now wait 24–48h for re-indexing.\n"
        "watch_index.py (cron every 2h) will detect the re-index and auto-run eval + decide."
    )


@cli.command()
@click.option("--k", default=None, type=int)
@click.option("--force", is_flag=True)
def start(k: int | None, force: bool) -> None:
    """
    Full setup: generate prompts + run baseline for all products.
    Run this once before starting mutation cycles.
    """
    ids = [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]

    # Step 1: Generate prompts (skip if already exist)
    click.echo("\n── Step 1: Generate prompts ─────────────────────────────")
    for pid in ids:
        prompts_path = config.PROMPTS_DIR / f"{pid}.yaml"
        if not force and prompts_path.exists():
            click.echo(f"  {pid}: prompts already exist — skip (use --force to regenerate)")
            continue
        run_script("prompts_gen.py", ["--product-id", pid, *(["--overwrite"] if force else [])])

    # Step 2: Run baseline
    click.echo("\n── Step 2: Run baseline evaluation ──────────────────────")
    args = ["--force"] if force else []
    if k:
        args += ["--k", str(k)]
    run_script("run_baseline.py", args)

    click.echo(
        "\n✓ Setup complete. Run `python pipeline.py status` to verify.\n"
        "Then run `python pipeline.py run` to start the first mutation cycle."
    )


@cli.command()
@click.option("--watch-interval", default=120, show_default=True, help="Check interval in minutes for cron setup guidance")
def install_cron(watch_interval: int) -> None:
    """Print the cron line to install watch_index.py."""
    cron_line = (
        f"0 */{watch_interval // 60 or 1} * * * "
        f"cd {config.ROOT} && "
        f"{sys.executable} watch_index.py >> {config.ROOT}/logs/watch.log 2>&1"
    )
    click.echo("Add this line to your crontab (crontab -e):\n")
    click.echo(cron_line)
    click.echo(
        "\nOr run manually: python watch_index.py\n"
        "(Re-index typically takes 24-48h — hourly cron is sufficient)"
    )


if __name__ == "__main__":
    cli()
