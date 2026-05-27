"""
config.py — Central configuration loader for catalog-optimizer.

All settings come from environment variables (loaded from .env if present).
No secrets are hard-coded; this file is safe to commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (no-op if running in a real environment)
load_dotenv(Path(__file__).parent / ".env")

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
PRODUCTS_DIR = ROOT / "products"
PROMPTS_DIR = ROOT / "prompts"
STATE_DIR = ROOT / "state"
EVALS_DIR = ROOT / "evals"

for _d in (PRODUCTS_DIR, PROMPTS_DIR, STATE_DIR, EVALS_DIR):
    _d.mkdir(exist_ok=True)

# ─── Shopify Admin API ────────────────────────────────────────────────────────

SHOPIFY_SHOP_DOMAIN: str = os.environ.get("SHOPIFY_SHOP_DOMAIN", "")
SHOPIFY_ADMIN_TOKEN: str = os.environ.get("SHOPIFY_ADMIN_TOKEN", "")
SHOPIFY_API_VERSION: str = os.environ.get("SHOPIFY_API_VERSION", "2025-01")

def shopify_graphql_url() -> str:
    if not SHOPIFY_SHOP_DOMAIN:
        raise EnvironmentError("SHOPIFY_SHOP_DOMAIN is not set")
    return f"https://{SHOPIFY_SHOP_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

def shopify_headers() -> dict[str, str]:
    if not SHOPIFY_ADMIN_TOKEN:
        raise EnvironmentError("SHOPIFY_ADMIN_TOKEN is not set")
    return {
        "X-Shopify-Access-Token": SHOPIFY_ADMIN_TOKEN,
        "Content-Type": "application/json",
    }

# ─── Shopify Catalog MCP ──────────────────────────────────────────────────────

CATALOG_MCP_URL: str = os.environ.get(
    "CATALOG_MCP_URL",
    "https://catalog.shopify.com/api/ucp/mcp",
)
CATALOG_API_KEY: str = os.environ.get("CATALOG_API_KEY", "")

# Use storefront-scoped endpoint if configured (single-store experiments)
STOREFRONT_MCP_URL: str = os.environ.get("STOREFRONT_MCP_URL", "")

def catalog_mcp_url() -> str:
    return STOREFRONT_MCP_URL or CATALOG_MCP_URL

def catalog_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if CATALOG_API_KEY:
        headers["Authorization"] = f"Bearer {CATALOG_API_KEY}"
    return headers

# ─── Anthropic ────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-7")

# ─── Experiment knobs ─────────────────────────────────────────────────────────

EVAL_K: int = int(os.environ.get("EVAL_K", "5"))
CYCLE_MIN_HOURS: int = int(os.environ.get("CYCLE_MIN_HOURS", "24"))
DECISION_SIGMA: float = float(os.environ.get("DECISION_SIGMA", "2.0"))

# Mutation classes in priority order (F is the control — run last)
MUTATION_ORDER: list[str] = ["A", "B", "C", "D", "E", "F"]

MUTATION_DESCRIPTIONS: dict[str, str] = {
    "A": "Title fitment — append explicit vehicle compat to title",
    "B": "Description fitment table — embed year/make/model/trim table in first 6000 chars",
    "C": "Description use-case language — buyer-intent phrasing in description",
    "D": "Tags — vehicle-specific tags (year ranges, generations, trims)",
    "E": "Standard Product Taxonomy — set Shopify standard product category",
    "F": "Metafield-only fitment — CONTROL: fitment ONLY in metafields, removed from title/description",
}

MUTATION_FIELDS: dict[str, str] = {
    "A": "title",
    "B": "descriptionHtml",
    "C": "descriptionHtml",
    "D": "tags",
    "E": "productCategory",
    "F": "metafields",
}

PROMPT_TIERS: list[str] = ["head", "mid", "long_tail", "fitment"]

# ─── Validation helpers ───────────────────────────────────────────────────────

def require_shopify_credentials() -> None:
    missing = []
    if not SHOPIFY_SHOP_DOMAIN:
        missing.append("SHOPIFY_SHOP_DOMAIN")
    if not SHOPIFY_ADMIN_TOKEN:
        missing.append("SHOPIFY_ADMIN_TOKEN")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example to .env and fill in your credentials."
        )

def require_anthropic_credentials() -> None:
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example to .env and fill in your API key."
        )
