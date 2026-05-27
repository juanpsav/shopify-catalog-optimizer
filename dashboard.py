"""
dashboard.py — Per-product mutation timeline and fitment-tier MRR progress chart.

Reads all eval results and history.jsonl files and renders:
  1. Terminal: rich table + ASCII sparkline per product
  2. HTML export: full interactive timeline (--html flag)

Usage:
    # Terminal dashboard
    python dashboard.py

    # Export HTML report
    python dashboard.py --html evals/dashboard.html

    # Show only one product
    python dashboard.py --product-id tacoma-bed-organizer
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

import config
from eval import load_all_evals
from mutate import load_history


# ─── ASCII sparkline ──────────────────────────────────────────────────────────

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as an ASCII sparkline."""
    if not values:
        return "─" * width
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v or 1e-9

    def char_for(v: float) -> str:
        idx = int((v - min_v) / span * (len(SPARK_CHARS) - 1))
        return SPARK_CHARS[min(idx, len(SPARK_CHARS) - 1)]

    return "".join(char_for(v) for v in values[-width:])


# ─── Per-product summary ──────────────────────────────────────────────────────

def build_product_timeline(product_id: str) -> dict[str, Any]:
    """
    Build a timeline dict for one product combining eval history and mutation history.
    """
    evals = load_all_evals(product_id)
    history = load_history(product_id)

    # Index history by cycle
    history_by_cycle: dict[int, dict] = {e.get("cycle", 0): e for e in history}

    timeline: list[dict[str, Any]] = []
    for ev in evals:
        label = ev.get("label", "eval")
        is_baseline = label.startswith("baseline")
        fitment_mrr = ev.get("fitment_mrr", 0.0)

        # Find matching history entry (by label parsing or chronological order)
        mutation_class = None
        decision = None
        hypothesis = None
        if not is_baseline:
            # Label format: mutation-A-cycle3
            parts = label.split("-")
            for p in parts:
                if len(p) == 1 and p.upper() in config.MUTATION_ORDER:
                    mutation_class = p.upper()
                    break
            for entry in history:
                if entry.get("mutation_class") == mutation_class:
                    decision = entry.get("decision") or entry.get("status")
                    hypothesis = entry.get("hypothesis")
                    break

        tiers = ev.get("tiers", {})
        timeline.append({
            "timestamp": ev.get("evaluated_at", ""),
            "label": label,
            "is_baseline": is_baseline,
            "mutation_class": mutation_class,
            "decision": decision,
            "hypothesis": hypothesis,
            "fitment_mrr": fitment_mrr,
            "head_mrr": tiers.get("head", {}).get("mrr", 0.0),
            "mid_mrr": tiers.get("mid", {}).get("mrr", 0.0),
            "long_tail_mrr": tiers.get("long_tail", {}).get("mrr", 0.0),
        })

    baseline_mrr = 0.0
    if timeline:
        baseline_evals = [t for t in timeline if t["is_baseline"]]
        if baseline_evals:
            baseline_mrr = baseline_evals[-1]["fitment_mrr"]

    # Compute lift vs baseline
    for point in timeline:
        if not point["is_baseline"] and baseline_mrr > 1e-9:
            point["lift_vs_baseline"] = (point["fitment_mrr"] - baseline_mrr) / baseline_mrr
        else:
            point["lift_vs_baseline"] = 0.0

    # Best mutation
    kept = [t for t in timeline if t.get("decision") == "KEEP"]
    best_mutation = max(kept, key=lambda t: t["fitment_mrr"]) if kept else None

    return {
        "product_id": product_id,
        "timeline": timeline,
        "baseline_mrr": baseline_mrr,
        "latest_mrr": timeline[-1]["fitment_mrr"] if timeline else 0.0,
        "best_mutation": best_mutation,
        "eval_count": len(timeline),
        "mrr_series": [t["fitment_mrr"] for t in timeline],
    }


# ─── Terminal rendering ───────────────────────────────────────────────────────

def render_terminal(product_summaries: list[dict[str, Any]]) -> None:
    try:
        from rich.table import Table
        from rich.console import Console
        from rich.panel import Panel
        from rich import box
        from rich.text import Text

        console = Console()

        console.print(Panel.fit(
            "[bold cyan]Catalog Optimizer — Fitment-Tier MRR Dashboard[/]\n"
            f"[dim]{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/]",
            border_style="cyan",
        ))

        # Main summary table
        table = Table(box=box.ROUNDED, show_footer=True)
        table.add_column("Product", style="cyan", footer="Total")
        table.add_column("Baseline MRR", justify="right")
        table.add_column("Latest MRR", justify="right")
        table.add_column("Best Lift", justify="right")
        table.add_column("Best Mutation", justify="center")
        table.add_column("MRR Timeline", justify="left", min_width=22)
        table.add_column("Evals", justify="right")

        total_evals = 0
        for s in product_summaries:
            baseline = s["baseline_mrr"]
            latest = s["latest_mrr"]
            best_mut = s.get("best_mutation")
            best_lift = best_mut.get("lift_vs_baseline", 0.0) if best_mut else 0.0
            best_cls = best_mut.get("mutation_class", "—") if best_mut else "—"
            spark = sparkline(s["mrr_series"])
            total_evals += s["eval_count"]

            lift_str = f"+{best_lift:.0%}" if best_lift > 0 else "—"
            latest_color = "green" if latest > baseline else "red" if latest < baseline else "white"

            table.add_row(
                s["product_id"],
                f"{baseline:.4f}",
                f"[{latest_color}]{latest:.4f}[/]",
                f"[{'green' if best_lift > 0 else 'dim'}]{lift_str}[/]",
                f"[bold]{best_cls}[/]",
                spark,
                str(s["eval_count"]),
            )

        table.columns[-1].footer = str(total_evals)
        console.print(table)

        # Per-product mutation log
        for s in product_summaries:
            if not s["timeline"]:
                continue
            console.print(f"\n[bold]{s['product_id']}[/] — mutation log")
            for point in s["timeline"]:
                icon = "🏁" if point["is_baseline"] else {
                    "KEEP": "✅",
                    "REVERT": "↩️",
                    "INCONCLUSIVE": "🔲",
                    "PENDING": "⏳",
                }.get(point.get("decision") or "PENDING", "▶")
                ts = (point["timestamp"] or "")[:10]
                cls_str = f"[{point['mutation_class']}] " if point.get("mutation_class") else ""
                dec_str = f" → {point['decision']}" if point.get("decision") else ""
                lift_str = (
                    f" (+{point['lift_vs_baseline']:.0%})" if point.get("lift_vs_baseline", 0) > 0
                    else ""
                )
                console.print(
                    f"  {icon} {ts}  {cls_str}fitment-MRR: {point['fitment_mrr']:.4f}{lift_str}{dec_str}"
                )
                if point.get("hypothesis") and point.get("decision") == "KEEP":
                    console.print(f"     [dim italic]{point['hypothesis'][:80]}…[/]")

    except ImportError:
        # Plain text fallback
        for s in product_summaries:
            print(f"\n{s['product_id']}")
            print(f"  Baseline MRR: {s['baseline_mrr']:.4f}")
            print(f"  Latest MRR:   {s['latest_mrr']:.4f}")
            print(f"  Sparkline:    {sparkline(s['mrr_series'])}")
            for point in s["timeline"]:
                ts = (point["timestamp"] or "")[:10]
                print(f"  {ts}  {point['label']:30s}  fitment={point['fitment_mrr']:.4f}  {point.get('decision') or 'pending'}")


# ─── HTML export ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Catalog Optimizer Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0; padding: 20px; background: #0f0f0f; color: #e0e0e0; }}
    h1 {{ color: #6ee7b7; margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 32px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 24px; }}
    .card {{ background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 12px;
             padding: 20px; }}
    .card h2 {{ font-size: 1rem; color: #a5f3fc; margin: 0 0 4px; }}
    .stats {{ display: flex; gap: 16px; margin: 8px 0 16px; }}
    .stat {{ text-align: center; }}
    .stat .val {{ font-size: 1.5rem; font-weight: 700; }}
    .stat .lbl {{ font-size: 0.75rem; color: #9ca3af; }}
    .green {{ color: #4ade80; }}
    .red {{ color: #f87171; }}
    .gray {{ color: #6b7280; }}
    canvas {{ max-height: 200px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85rem; }}
    th, td {{ padding: 4px 8px; text-align: left; border-bottom: 1px solid #2d2d2d; }}
    th {{ color: #6b7280; font-weight: normal; }}
    .keep {{ color: #4ade80; }} .revert {{ color: #f87171; }}
    .inconclusive {{ color: #facc15; }} .pending {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Catalog Optimizer — Fitment-Tier MRR Dashboard</h1>
  <p class="subtitle">Generated {generated_at}</p>
  <div class="grid">
    {cards}
  </div>
  <script>
    {chart_scripts}
  </script>
</body>
</html>
"""

def render_html(product_summaries: list[dict[str, Any]], output_path: Path) -> None:
    cards = []
    chart_scripts = []

    for s in product_summaries:
        pid = s["product_id"]
        safe_id = pid.replace("-", "_").replace(".", "_")
        baseline = s["baseline_mrr"]
        latest = s["latest_mrr"]
        delta = latest - baseline
        color_class = "green" if delta > 0 else "red" if delta < 0 else "gray"

        # Build chart data
        labels = [t["label"] for t in s["timeline"]]
        fitment_data = [t["fitment_mrr"] for t in s["timeline"]]
        head_data = [t["head_mrr"] for t in s["timeline"]]
        mid_data = [t["mid_mrr"] for t in s["timeline"]]

        # Mutation table rows
        mut_rows = []
        for point in s["timeline"]:
            dec = point.get("decision") or ("baseline" if point["is_baseline"] else "pending")
            dec_class = dec.lower() if dec else ""
            hyp = (point.get("hypothesis") or "")[:60]
            mut_rows.append(
                f"<tr><td>{point['timestamp'][:10]}</td>"
                f"<td>{point.get('mutation_class') or '—'}</td>"
                f"<td>{point['fitment_mrr']:.4f}</td>"
                f"<td class='{dec_class}'>{dec}</td>"
                f"<td>{hyp}</td></tr>"
            )

        card_html = f"""
    <div class="card">
      <h2>{pid}</h2>
      <div class="stats">
        <div class="stat"><div class="val gray">{baseline:.4f}</div><div class="lbl">Baseline MRR</div></div>
        <div class="stat"><div class="val {color_class}">{latest:.4f}</div><div class="lbl">Latest MRR</div></div>
        <div class="stat"><div class="val {color_class}">{delta:+.4f}</div><div class="lbl">Delta</div></div>
        <div class="stat"><div class="val">{s['eval_count']}</div><div class="lbl">Evals</div></div>
      </div>
      <canvas id="chart_{safe_id}"></canvas>
      <table>
        <tr><th>Date</th><th>Class</th><th>Fitment MRR</th><th>Decision</th><th>Hypothesis</th></tr>
        {''.join(mut_rows)}
      </table>
    </div>"""
        cards.append(card_html)

        chart_script = f"""
    (function() {{
      const ctx = document.getElementById('chart_{safe_id}').getContext('2d');
      new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: {json.dumps(labels)},
          datasets: [
            {{ label: 'Fitment MRR', data: {json.dumps(fitment_data)},
               borderColor: '#6ee7b7', backgroundColor: 'rgba(110,231,183,0.1)',
               borderWidth: 2, tension: 0.3, fill: true }},
            {{ label: 'Head MRR', data: {json.dumps(head_data)},
               borderColor: '#818cf8', borderWidth: 1, tension: 0.3 }},
            {{ label: 'Mid MRR', data: {json.dumps(mid_data)},
               borderColor: '#fb923c', borderWidth: 1, tension: 0.3 }},
          ]
        }},
        options: {{
          responsive: true, plugins: {{ legend: {{ labels: {{ color: '#e0e0e0' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#6b7280' }}, grid: {{ color: '#2d2d2d' }} }},
            y: {{ min: 0, max: 1, ticks: {{ color: '#6b7280' }}, grid: {{ color: '#2d2d2d' }} }}
          }}
        }}
      }});
    }})();"""
        chart_scripts.append(chart_script)

    html = HTML_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        cards="\n".join(cards),
        chart_scripts="\n".join(chart_scripts),
    )
    output_path.write_text(html, encoding="utf-8")
    click.echo(f"HTML dashboard written: {output_path}")


# ─── Field-impact table ───────────────────────────────────────────────────────

def render_field_impact_table(product_summaries: list[dict[str, Any]]) -> None:
    """
    Print the field-impact table: which mutation class moved fitment MRR the most,
    averaged across all products. This is the headline deliverable.
    """
    # Collect per-class lift values across all products
    class_lifts: dict[str, list[float]] = {cls: [] for cls in config.MUTATION_ORDER}

    for s in product_summaries:
        baseline = s["baseline_mrr"]
        for point in s["timeline"]:
            cls = point.get("mutation_class")
            if cls and not point["is_baseline"] and point.get("decision") in ("KEEP", "REVERT", "INCONCLUSIVE"):
                lift = point["fitment_mrr"] - baseline
                class_lifts[cls].append(lift)

    try:
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()
        table = Table(
            box=box.ROUNDED,
            title="Field Impact on Fitment-Tier MRR",
            caption="↑ = positive lift. F is the control (metafield-only — expected near 0).",
        )
        table.add_column("Class", justify="center", style="bold")
        table.add_column("Field / Strategy", style="cyan")
        table.add_column("Avg Lift", justify="right")
        table.add_column("Products Tested", justify="right")
        table.add_column("Expected Lift", justify="center")

        expected_lift = {"A": "High", "B": "High", "C": "Med-High", "D": "Medium", "E": "Medium", "F": "CONTROL ≈0"}
        for cls in config.MUTATION_ORDER:
            lifts = class_lifts[cls]
            n = len(lifts)
            avg = statistics.mean(lifts) if lifts else None
            avg_str = f"{avg:+.4f}" if avg is not None else "—"
            color = "green" if avg and avg > 0 else "red" if avg and avg < 0 else "dim"
            table.add_row(
                cls,
                config.MUTATION_DESCRIPTIONS[cls].split(" — ")[0][:45],
                f"[{color}]{avg_str}[/]",
                str(n),
                expected_lift[cls],
            )

        console.print("\n")
        console.print(table)
    except ImportError:
        for cls in config.MUTATION_ORDER:
            lifts = class_lifts[cls]
            avg = statistics.mean(lifts) if lifts else None
            click.echo(f"  {cls}: {avg:+.4f}" if avg is not None else f"  {cls}: no data")


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", default=None, help="Show only this product")
@click.option("--html", "html_output", default=None, type=click.Path(), help="Export HTML dashboard")
@click.option("--field-impact", is_flag=True, help="Show field-impact table (final deliverable)")
def main(product_id: str | None, html_output: str | None, field_impact: bool) -> None:
    """Terminal dashboard + optional HTML export of fitment-tier MRR timelines."""

    ids = (
        [product_id]
        if product_id
        else [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]
    )

    summaries = [build_product_timeline(pid) for pid in ids]

    render_terminal(summaries)

    if field_impact:
        render_field_impact_table(summaries)

    if html_output:
        render_html(summaries, Path(html_output))


if __name__ == "__main__":
    main()
