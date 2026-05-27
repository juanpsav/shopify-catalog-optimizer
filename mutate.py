"""
mutate.py — Mutation engine: proposes one field delta per cycle and applies it.

Mutation priority order: A → B → C → D → E → F
One mutation per product per cycle. After applying, writes:
  - products/<id>/pending_index.json  (what to look for after re-index)
  - an entry to products/<id>/history.jsonl

Usage:
    # Propose and apply the next mutation for one product
    python mutate.py --product-id tacoma-bed-organizer

    # Dry-run: propose but don't call Admin API
    python mutate.py --product-id tacoma-bed-organizer --dry-run

    # Force a specific mutation class
    python mutate.py --product-id tacoma-bed-organizer --class A
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import click

import config
from admin_client import get_admin_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── History helpers ──────────────────────────────────────────────────────────

def load_history(product_id: str) -> list[dict[str, Any]]:
    path = config.PRODUCTS_DIR / product_id / "history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    result = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def append_history(product_id: str, entry: dict[str, Any]) -> None:
    path = config.PRODUCTS_DIR / product_id / "history.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_history_entry(product_id: str, cycle: int, updates: dict[str, Any]) -> None:
    """Update a specific history entry (by cycle number) in place."""
    path = config.PRODUCTS_DIR / product_id / "history.jsonl"
    if not path.exists():
        return
    lines = path.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("cycle") == cycle:
                entry.update(updates)
            new_lines.append(json.dumps(entry, ensure_ascii=False))
        except json.JSONDecodeError:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n")


# ─── Cycle tracking ───────────────────────────────────────────────────────────

def next_mutation_class(history: list[dict[str, Any]]) -> str | None:
    """
    Return the next mutation class to apply, or None if all have been tried.

    Skips classes that already have a KEEP or INCONCLUSIVE entry.
    Re-runs classes where the result was REVERT (to re-test after revert).
    Never applies a class currently PENDING.
    """
    applied: dict[str, str] = {}  # class → status
    for entry in history:
        cls = entry.get("mutation_class", "")
        status = entry.get("status", "")
        if cls and status:
            applied[cls] = status

    for cls in config.MUTATION_ORDER:
        status = applied.get(cls)
        if status is None:
            return cls  # Not tried yet
        if status == "PENDING":
            return None  # Wait for current cycle to complete
        # KEEP / INCONCLUSIVE / REVERT → move to next class
    return None  # All classes exhausted


def current_cycle_number(history: list[dict[str, Any]]) -> int:
    if not history:
        return 1
    return max(e.get("cycle", 0) for e in history) + 1


# ─── Mutation proposal (LLM) ─────────────────────────────────────────────────

MUTATION_SYSTEM_PROMPT = """\
You are an expert Shopify product content strategist specializing in fitment-heavy
automotive accessory catalogs. Your goal is to craft a specific product field mutation
that maximizes the product's visibility in AI-powered catalog search (ChatGPT Shopping,
Google Gemini, Shopify Shop assistant).

The key insight: fitment signal (year/make/model/trim) must live in natural-language
text fields (title, description) — NOT only in structured metafields — to rank in
AI catalog search. Your mutations encode vehicle compatibility into the right fields.

You will be given:
- mutation_class: which field to change and the strategy
- current product data
- fitment vehicle compatibility list
- mutation history (what has already been tried)

Respond ONLY with a JSON object with these keys:
  field: string (the Shopify Admin API field name to update)
  new_value: the new field value (string, array, or object as appropriate)
  prior_value: the current value of that field (copy from current product data)
  hypothesis: 1-2 sentences explaining why this specific change should improve
              fitment-tier catalog ranking

For descriptionHtml mutations, output valid HTML. Keep total description under 6000
characters. For title mutations, keep under 80 characters. For tags, output an array
of strings. For metafields, output an array of metafield objects.

No explanation outside the JSON. No markdown fences.
"""

MUTATION_USER_TEMPLATE = """\
Mutation class: {mutation_class}
Class description: {class_description}

Current product:
{current_json}

Fitment data:
{fitment_json}

Previous mutations (for context):
{history_json}

Generate the mutation.
"""

# Class-specific guidance injected into the system prompt
CLASS_GUIDANCE: dict[str, str] = {
    "A": (
        "Title fitment: Append the most important vehicle compatibility to the title. "
        "Format: '<existing title> for <year_start>-<year_end> <Make> <Model> (<Generation>)'. "
        "Keep under 80 chars. Use the first/most popular vehicle from fitment.json. "
        "Field to set: 'title'."
    ),
    "B": (
        "Description fitment table: Add a structured vehicle compatibility table in the first "
        "paragraph of the description. Use an HTML table with columns: Year, Make, Model, Trim. "
        "List every compatible vehicle. Place this table BEFORE any existing description text. "
        "Field to set: 'descriptionHtml'. Keep total under 6000 chars."
    ),
    "C": (
        "Description use-case language: Add buyer-intent paragraphs to the description. "
        "Include: use-case scenarios (overlanding, towing, daily driver), vehicle-specific "
        "benefit statements, and natural language fitment mentions. "
        "Append AFTER existing description. Field: 'descriptionHtml'."
    ),
    "D": (
        "Tags: Add vehicle-specific tags. Include: year ranges (e.g. '2022-2024'), "
        "generation names (e.g. '3rd-gen-tacoma'), make+model combos (e.g. 'toyota-tacoma'), "
        "trim names (e.g. 'trd-pro'), and use-case tags. "
        "PRESERVE all existing tags. Field: 'tags' (full array including existing)."
    ),
    "E": (
        "Standard Product Taxonomy: Set the Shopify standard product category using the "
        "'productCategory' field. Use the most specific Shopify standard taxonomy category "
        "that fits this product (e.g. 'Vehicles & Parts > Vehicle Parts & Accessories > "
        "Motor Vehicle Cargo & Accessories'). Field: 'productType' (use descriptive string)."
    ),
    "F": (
        "CONTROL — Metafield-only fitment: REVERT the title and description to versions that "
        "contain NO vehicle-specific text. Move ALL fitment data into a single "
        "'fitment.compatible_vehicles' metafield as JSON. This tests whether structured "
        "metafields alone carry catalog ranking signal. Field: 'metafields' + reset title/description."
    ),
}


def propose_mutation(
    product_id: str,
    mutation_class: str,
    current: dict[str, Any],
    fitment: dict[str, Any],
    history: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    """
    Use Claude to propose a specific field mutation.
    Returns dict with: field, new_value, prior_value, hypothesis.
    """
    config.require_anthropic_credentials()
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Build system prompt with class-specific guidance
    system = MUTATION_SYSTEM_PROMPT + "\n\nClass-specific guidance:\n" + CLASS_GUIDANCE[mutation_class]

    # Trim description to reduce tokens
    trimmed = dict(current)
    desc = trimmed.get("description_html", "")
    if len(desc) > 1500:
        trimmed["description_html"] = desc[:1500] + "…"

    user_message = MUTATION_USER_TEMPLATE.format(
        mutation_class=mutation_class,
        class_description=config.MUTATION_DESCRIPTIONS[mutation_class],
        current_json=json.dumps(trimmed, indent=2),
        fitment_json=json.dumps(fitment, indent=2),
        history_json=json.dumps(
            [{"cycle": e.get("cycle"), "class": e.get("mutation_class"),
              "status": e.get("status"), "hypothesis": e.get("hypothesis")} for e in history],
            indent=2,
        ),
    )

    response = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()

    proposal = json.loads(raw)

    # Validate required keys
    for key in ("field", "new_value", "hypothesis"):
        if key not in proposal:
            raise ValueError(f"Mutation proposal missing key: {key!r}")

    return proposal


# ─── Apply mutation ───────────────────────────────────────────────────────────

def apply_mutation(
    product_id: str,
    product_gid: str,
    proposal: dict[str, Any],
    mutation_class: str,
    cycle: int,
    dry_run: bool = False,
) -> None:
    """
    Apply the proposed mutation via Admin API, then write
    pending_index.json and a history entry.
    """
    field = proposal["field"]
    new_value = proposal["new_value"]
    prior_value = proposal.get("prior_value")
    hypothesis = proposal["hypothesis"]
    now = datetime.now(timezone.utc).isoformat()

    # Admin API field mapping (proposal may use our internal names)
    api_field_map = {
        "description_html": "descriptionHtml",
        "product_type": "productType",
        "taxonomy_category": "productType",  # closest writable analog
    }
    api_field = api_field_map.get(field, field)

    if not dry_run:
        admin = get_admin_client()
        admin.update_product(product_gid, {api_field: new_value})
        click.echo(f"  Applied {mutation_class} mutation to {product_id} (field: {api_field})")
    else:
        click.echo(f"  [DRY-RUN] Would apply {mutation_class} to {product_id}: {api_field} = {str(new_value)[:120]}…")

    # Write pending_index.json
    pending_path = config.PRODUCTS_DIR / product_id / "pending_index.json"
    pending = {
        "mutation_cycle": cycle,
        "mutation_class": mutation_class,
        "field": api_field,
        "expected_value": new_value,
        "written_at": now,
        "min_check_after": _hours_later(now, config.CYCLE_MIN_HOURS),
        "dry_run": dry_run,
    }
    pending_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False))

    # Write history entry
    history_entry = {
        "timestamp": now,
        "cycle": cycle,
        "mutation_class": mutation_class,
        "field": api_field,
        "hypothesis": hypothesis,
        "prior_value": prior_value,
        "new_value": new_value,
        "status": "PENDING",
        "eval_score": None,
        "baseline_score": None,
        "decision": None,
        "dry_run": dry_run,
    }
    append_history(product_id, history_entry)
    click.echo(f"  History entry written (cycle {cycle}, status=PENDING)")
    click.echo(f"  Hypothesis: {hypothesis}")


def _hours_later(iso_ts: str, hours: int) -> str:
    from datetime import timedelta
    dt = datetime.fromisoformat(iso_ts)
    return (dt + timedelta(hours=hours)).isoformat()


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--product-id", required=True, help="Local product ID (products/ subdir name)")
@click.option("--class", "mutation_class", default=None,
              help="Force a specific mutation class (A-F). Default: auto-select next.")
@click.option("--dry-run", is_flag=True, help="Propose mutation but don't call Admin API")
@click.option("--model", default=None, help="Claude model override")
def main(
    product_id: str,
    mutation_class: str | None,
    dry_run: bool,
    model: str | None,
) -> None:
    """Propose and apply the next mutation for a product."""

    product_dir = config.PRODUCTS_DIR / product_id
    if not product_dir.is_dir():
        click.echo(f"No products/{product_id}/ directory found.", err=True)
        sys.exit(1)

    current = json.loads((product_dir / "current.json").read_text())
    fitment = json.loads((product_dir / "fitment.json").read_text())
    history = load_history(product_id)

    # Check for already-pending mutation
    pending_path = product_dir / "pending_index.json"
    if pending_path.exists() and not dry_run:
        pending = json.loads(pending_path.read_text())
        click.echo(
            f"Product {product_id} already has a PENDING mutation "
            f"(class {pending.get('mutation_class')}, cycle {pending.get('mutation_cycle')}). "
            f"Wait for re-index (min: {pending.get('min_check_after')}) before next mutation."
        )
        sys.exit(0)

    # Determine mutation class
    if mutation_class:
        cls = mutation_class.upper()
        if cls not in config.MUTATION_ORDER:
            click.echo(f"Invalid mutation class {cls!r}. Choose from: {config.MUTATION_ORDER}", err=True)
            sys.exit(1)
    else:
        cls = next_mutation_class(history)
        if cls is None:
            click.echo(f"No mutation classes remaining for {product_id}. All classes exhausted.")
            sys.exit(0)

    cycle = current_cycle_number(history)
    product_gid = current["product_id"]

    click.echo(f"\nMutation: {product_id} — Class {cls}: {config.MUTATION_DESCRIPTIONS[cls]}")
    click.echo(f"Cycle: {cycle}")
    click.echo("Calling Claude to propose mutation…")

    proposal = propose_mutation(product_id, cls, current, fitment, history, model=model)

    click.echo(f"\nProposed field: {proposal['field']}")
    click.echo(f"Hypothesis: {proposal['hypothesis']}")
    new_val_preview = str(proposal['new_value'])[:200]
    click.echo(f"New value (preview): {new_val_preview}{'…' if len(str(proposal['new_value'])) > 200 else ''}")

    apply_mutation(product_id, product_gid, proposal, cls, cycle, dry_run=dry_run)


if __name__ == "__main__":
    main()
