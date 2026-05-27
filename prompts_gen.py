"""
prompts_gen.py — LLM-based 4-tier fitment-aware prompt set generator.

For each product, reads fitment.json and current.json, then calls Claude
to generate a realistic prompt set covering four tiers:
  - head:      bare category queries ("truck bed organizer")
  - mid:       attribute-qualified queries ("aluminum truck bed drawer")
  - long_tail: use-case / buyer-intent queries
  - fitment:   vehicle-specific queries (the critical tier for this experiment)

Outputs: prompts/<product_id>.yaml

Usage:
    # Generate for a specific product
    python prompts_gen.py --product-id tacoma-bed-organizer

    # Regenerate for all products (overwrites existing)
    python prompts_gen.py --all --overwrite

    # Dry-run: print to stdout without writing
    python prompts_gen.py --product-id tacoma-bed-organizer --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import anthropic
import click
import yaml

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─── Prompt generation ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a search-query strategist for an automotive aftermarket parts experiment.
Your job is to generate realistic buyer search queries that someone would type into
ChatGPT, Google, or an AI shopping assistant to find a specific vehicle accessory.

Rules:
1. Head queries: 2-4 words, bare product category, no brand/vehicle mentions.
2. Mid queries: 5-8 words, include one key attribute (material, size, feature).
3. Long-tail queries: 8-15 words, buyer intent or use-case phrasing. Sound natural.
4. Fitment queries: vehicle-specific. Must include year range OR generation name
   AND make AND model. Use the exact vehicles from the fitment data provided.
   Generate at least one query per distinct vehicle entry.

Output ONLY a JSON object with keys: head, mid, long_tail, fitment.
Each key maps to a list of strings. Aim for 3-5 queries per tier.
No markdown. No explanation. Just the JSON.
"""

USER_TEMPLATE = """\
Product:
{current_json}

Fitment data:
{fitment_json}

Generate the 4-tier prompt set for this product.
"""


def generate_prompts(
    product_id: str,
    current: dict[str, Any],
    fitment: dict[str, Any],
    model: str | None = None,
) -> dict[str, list[str]]:
    """
    Call Claude to generate the 4-tier prompt set.
    Returns a dict with keys: head, mid, long_tail, fitment.
    """
    config.require_anthropic_credentials()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Trim description to avoid token waste
    trimmed_current = dict(current)
    desc = trimmed_current.get("description_html", "")
    if len(desc) > 800:
        trimmed_current["description_html"] = desc[:800] + "…"

    user_message = USER_TEMPLATE.format(
        current_json=json.dumps(trimmed_current, indent=2),
        fitment_json=json.dumps(fitment, indent=2),
    )

    response = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw_text.startswith("```"):
        raw_text = "\n".join(
            line for line in raw_text.splitlines()
            if not line.startswith("```")
        ).strip()

    parsed = json.loads(raw_text)

    # Ensure all tiers are present (fall back to empty list)
    result: dict[str, list[str]] = {}
    for tier in config.PROMPT_TIERS:
        result[tier] = [str(q) for q in parsed.get(tier, [])]

    return result


def write_prompts_yaml(product_id: str, tiers: dict[str, list[str]]) -> Path:
    """Write prompts/<product_id>.yaml."""
    out_path = config.PROMPTS_DIR / f"{product_id}.yaml"
    data = {"product_id": product_id, "tiers": tiers}
    out_path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", out_path)
    return out_path


def load_prompts(product_id: str) -> dict[str, list[str]]:
    """Load prompts/<product_id>.yaml. Returns empty tiers if not found."""
    path = config.PROMPTS_DIR / f"{product_id}.yaml"
    if not path.exists():
        return {tier: [] for tier in config.PROMPT_TIERS}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tiers", {tier: [] for tier in config.PROMPT_TIERS})


def list_product_ids() -> list[str]:
    return [d.name for d in sorted(config.PRODUCTS_DIR.iterdir()) if d.is_dir()]


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", default=None, help="Single product to generate prompts for")
@click.option("--all", "all_products", is_flag=True, help="Generate for all products")
@click.option("--overwrite", is_flag=True, help="Overwrite existing prompt files")
@click.option("--dry-run", is_flag=True, help="Print YAML to stdout without writing")
@click.option("--model", default=None, help="Claude model override")
def main(
    product_id: str | None,
    all_products: bool,
    overwrite: bool,
    dry_run: bool,
    model: str | None,
) -> None:
    """Generate 4-tier fitment-aware prompt sets via Claude."""

    if not product_id and not all_products:
        click.echo("Pass --product-id <id> or --all. Run --help for usage.")
        sys.exit(1)

    ids = list_product_ids() if all_products else [product_id]

    for pid in ids:
        product_dir = config.PRODUCTS_DIR / pid
        if not product_dir.is_dir():
            click.echo(f"No products/{pid}/ directory found — skipping", err=True)
            continue

        out_path = config.PROMPTS_DIR / f"{pid}.yaml"
        if not overwrite and not dry_run and out_path.exists():
            click.echo(f"  skip {pid} (prompt file exists, use --overwrite to regenerate)")
            continue

        current_path = product_dir / "current.json"
        fitment_path = product_dir / "fitment.json"
        if not current_path.exists() or not fitment_path.exists():
            click.echo(f"  skip {pid} — missing current.json or fitment.json", err=True)
            continue

        current = json.loads(current_path.read_text())
        fitment = json.loads(fitment_path.read_text())

        click.echo(f"  Generating prompts for {pid}…")
        try:
            tiers = generate_prompts(pid, current, fitment, model=model)
        except Exception as exc:
            click.echo(f"  ERROR {pid}: {exc}", err=True)
            continue

        if dry_run:
            data = {"product_id": pid, "tiers": tiers}
            click.echo(
                yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            )
        else:
            write_prompts_yaml(pid, tiers)
            for tier, queries in tiers.items():
                click.echo(f"    {tier}: {len(queries)} queries")

    click.echo("Done.")


if __name__ == "__main__":
    main()
