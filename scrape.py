"""
scrape.py — Product data ingestion and normalization.

Pulls product data from one of three sources:
  1. Shopify Admin API (--shop-product-id)
  2. CSV / JSON export file (--file)
  3. Existing products/ directory (--scan — just validates and re-normalizes)

Outputs per-product:
  products/<product_id>/current.json   — current Shopify field values
  products/<product_id>/fitment.json   — normalized vehicle compat list

Usage:
    # Ingest a single product from the Admin API
    python scrape.py --shop-product-id gid://shopify/Product/123456789 \\
                     --local-id my-product-slug

    # Ingest from a CSV file (see docs/csv_schema.md for column names)
    python scrape.py --file products.csv

    # Ingest from a JSON array of product objects
    python scrape.py --file products.json

    # Re-validate all existing products/ directories
    python scrape.py --scan
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import click

import config
from admin_client import get_admin_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_current(raw_admin: dict[str, Any], local_id: str) -> dict[str, Any]:
    """Convert an Admin API product dict to products/<id>/current.json format."""
    return {
        "product_id": raw_admin.get("id", f"gid://shopify/Product/PLACEHOLDER_{local_id.upper()}"),
        "title": raw_admin.get("title", ""),
        "description_html": raw_admin.get("descriptionHtml", ""),
        "tags": raw_admin.get("tags", []),
        "product_type": raw_admin.get("productType", ""),
        "vendor": raw_admin.get("vendor", ""),
        "status": raw_admin.get("status", "ACTIVE"),
        "handle": raw_admin.get("handle", local_id),
        "metafields": raw_admin.get("metafields", []),
        "taxonomy_category": raw_admin.get("taxonomy_category"),
        "last_updated": raw_admin.get("updatedAt", ""),
    }


def normalize_fitment_from_metafield(current: dict[str, Any]) -> dict[str, Any]:
    """
    Try to extract fitment data from a compatible_vehicles metafield (if present).
    Returns a fitment.json-compatible dict.
    """
    vehicles: list[dict] = []
    for mf in current.get("metafields", []):
        if mf.get("key") in ("compatible_vehicles", "fitment", "vehicles", "aces_fitment"):
            try:
                value = json.loads(mf.get("value", "{}"))
                if isinstance(value, dict) and "vehicles" in value:
                    vehicles = value["vehicles"]
                elif isinstance(value, list):
                    vehicles = value
            except (json.JSONDecodeError, TypeError):
                pass
            break

    return {
        "product_id": current["product_id"],
        "vehicles": vehicles,
        "fitment_notes": "",
        "target_queries": [],
    }


# ─── CSV / JSON ingestion ─────────────────────────────────────────────────────

# Expected CSV columns (case-insensitive, extras are ignored)
CSV_COLUMNS = {
    "local_id", "shopify_product_id", "title", "description", "tags",
    "product_type", "vendor", "status", "make", "model", "year_start",
    "year_end", "trim", "generation", "fitment_notes",
}


def ingest_csv(path: Path) -> list[tuple[str, dict, dict]]:
    """
    Read a CSV file and return list of (local_id, current_dict, fitment_dict).

    CSV must have at minimum: local_id, title.
    Vehicle columns (make, model, year_start, year_end) produce one fitment vehicle entry.
    For multi-vehicle products, use one row per vehicle and duplicate the product columns.
    """
    results: list[tuple[str, dict, dict]] = []
    grouped: dict[str, list[dict]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize column names
            row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
            local_id = row.get("local_id") or row.get("id") or ""
            if not local_id:
                logger.warning("Skipping row without local_id: %s", row)
                continue
            grouped.setdefault(local_id, []).append(row)

    for local_id, rows in grouped.items():
        first = rows[0]
        current: dict[str, Any] = {
            "product_id": first.get("shopify_product_id")
                or f"gid://shopify/Product/PLACEHOLDER_{local_id.upper().replace('-', '_')}",
            "title": first.get("title", ""),
            "description_html": first.get("description", first.get("description_html", "")),
            "tags": [t.strip() for t in first.get("tags", "").split(",") if t.strip()],
            "product_type": first.get("product_type", ""),
            "vendor": first.get("vendor", ""),
            "status": first.get("status", "ACTIVE").upper(),
            "handle": first.get("handle", local_id),
            "metafields": [],
            "taxonomy_category": first.get("taxonomy_category"),
            "last_updated": "",
        }

        vehicles = []
        for row in rows:
            make = row.get("make", "")
            model = row.get("model", "")
            if make and model:
                vehicle: dict[str, Any] = {
                    "make": make,
                    "model": model,
                }
                for int_field in ("year_start", "year_end"):
                    val = row.get(int_field, "")
                    if val:
                        try:
                            vehicle[int_field] = int(val)
                        except ValueError:
                            pass
                for str_field in ("trim", "generation", "bed_length", "drivetrain", "notes"):
                    val = row.get(str_field, "")
                    if val:
                        # comma-separated → list
                        if "," in val:
                            vehicle[str_field] = [v.strip() for v in val.split(",")]
                        else:
                            vehicle[str_field] = val
                vehicles.append(vehicle)

        fitment: dict[str, Any] = {
            "product_id": current["product_id"],
            "vehicles": vehicles,
            "fitment_notes": first.get("fitment_notes", ""),
            "target_queries": [],
        }

        results.append((local_id, current, fitment))

    return results


def ingest_json(path: Path) -> list[tuple[str, dict, dict]]:
    """
    Read a JSON file (array of product objects) and return
    list of (local_id, current_dict, fitment_dict).

    Each object must have at least: local_id (or id), title.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    results: list[tuple[str, dict, dict]] = []
    for obj in data:
        local_id = obj.get("local_id") or obj.get("id") or obj.get("handle", "")
        if not local_id:
            logger.warning("Skipping JSON object without local_id: %s", list(obj.keys()))
            continue

        current = normalize_current(obj, local_id)
        fitment_data = obj.get("fitment", {})
        if fitment_data:
            fitment: dict[str, Any] = {
                "product_id": current["product_id"],
                "vehicles": fitment_data.get("vehicles", []),
                "fitment_notes": fitment_data.get("fitment_notes", ""),
                "target_queries": fitment_data.get("target_queries", []),
            }
        else:
            fitment = normalize_fitment_from_metafield(current)

        results.append((local_id, current, fitment))

    return results


# ─── Writer ───────────────────────────────────────────────────────────────────

def write_product_files(
    local_id: str,
    current: dict[str, Any],
    fitment: dict[str, Any],
    overwrite: bool = False,
) -> Path:
    """Write current.json and fitment.json into products/<local_id>/."""
    product_dir = config.PRODUCTS_DIR / local_id
    product_dir.mkdir(exist_ok=True)

    current_path = product_dir / "current.json"
    fitment_path = product_dir / "fitment.json"
    history_path = product_dir / "history.jsonl"

    if not overwrite and current_path.exists():
        logger.info("Skipping %s (already exists, use --overwrite to replace)", local_id)
        return product_dir

    current_path.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    fitment_path.write_text(json.dumps(fitment, indent=2, ensure_ascii=False))
    if not history_path.exists():
        history_path.write_text("")

    logger.info("Wrote products/%s/ (current.json + fitment.json)", local_id)
    return product_dir


def scan_existing() -> list[str]:
    """Validate existing products/ directories and return list of local IDs."""
    ids = []
    for d in sorted(config.PRODUCTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        current_ok = (d / "current.json").exists()
        fitment_ok = (d / "fitment.json").exists()
        if current_ok and fitment_ok:
            ids.append(d.name)
            logger.info("  ✓ %s", d.name)
        else:
            missing = []
            if not current_ok:
                missing.append("current.json")
            if not fitment_ok:
                missing.append("fitment.json")
            logger.warning("  ✗ %s — missing: %s", d.name, ", ".join(missing))
    return ids


# ─── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--shop-product-id", default=None, help="Shopify GID to pull from Admin API")
@click.option("--local-id", default=None, help="Local slug for products/ directory")
@click.option("--file", "input_file", default=None, type=click.Path(exists=True), help="CSV or JSON input file")
@click.option("--scan", is_flag=True, help="Validate existing products/ directories")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def main(
    shop_product_id: str | None,
    local_id: str | None,
    input_file: str | None,
    scan: bool,
    overwrite: bool,
) -> None:
    """Ingest product data into products/<id>/ for the catalog optimizer."""

    if scan:
        click.echo("Scanning products/ directories…")
        ids = scan_existing()
        click.echo(f"\nFound {len(ids)} valid product(s): {', '.join(ids)}")
        return

    if shop_product_id:
        if not local_id:
            # Derive local-id from numeric product ID
            match = re.search(r"(\d+)$", shop_product_id)
            local_id = f"product-{match.group(1)}" if match else "product-unknown"

        click.echo(f"Fetching {shop_product_id} from Admin API…")
        client = get_admin_client()
        raw = client.get_product(shop_product_id)
        current = normalize_current(raw, local_id)
        fitment = normalize_fitment_from_metafield(current)
        write_product_files(local_id, current, fitment, overwrite=overwrite)
        click.echo(f"Done. Review and edit products/{local_id}/fitment.json to add vehicle compat list.")
        return

    if input_file:
        path = Path(input_file)
        if path.suffix.lower() == ".csv":
            rows = ingest_csv(path)
        elif path.suffix.lower() == ".json":
            rows = ingest_json(path)
        else:
            click.echo(f"Unsupported file type: {path.suffix}. Use .csv or .json.", err=True)
            sys.exit(1)

        for lid, current, fitment in rows:
            write_product_files(lid, current, fitment, overwrite=overwrite)

        click.echo(f"Ingested {len(rows)} product(s) from {path.name}")
        return

    click.echo("Nothing to do. Pass --shop-product-id, --file, or --scan.")
    click.echo("Run 'python scrape.py --help' for usage.")


if __name__ == "__main__":
    main()
