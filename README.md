# Catalog Optimizer

An eval harness that measures which Shopify product fields actually move ranking for fitment-dependent products in AI-powered catalog search (ChatGPT Shopping, Gemini, Copilot, Shop assistant).

## The Problem

Fitment-heavy merchants (automotive parts, vehicle accessories) encode compatibility as structured data: metafields, ACES/PIES feeds, custom JSON. AI catalog search is not reading that. A part that technically fits a buyer's vehicle never surfaces because the year/make/model signal is not in the fields the ranking model reads.

This system quantifies the gap and closes it one field at a time.

## How It Works

For each product, the pipeline:

1. Runs a baseline: searches the Shopify Global Catalog with vehicle-specific queries and records where the product ranks (or whether it shows up at all)
2. Applies one field change per 24-48h cycle (title, description, tags, taxonomy, metafields)
3. Waits for re-indexing, then runs the same searches again
4. Compares the before and after rank to decide whether to keep the change or revert it

The control mutation (Class F) removes all fitment text from title and description and moves it entirely into metafields. If rank does not change, the experiment confirms that AI catalog search requires fitment signal in natural-language text, not structured metadata. That is the headline finding for fitment-heavy merchants.

## Mutation Classes

| Class | Field | Strategy |
|-------|-------|----------|
| A | title | Append explicit vehicle compat ("for 2024 Toyota Tacoma 3rd Gen") |
| B | descriptionHtml | Embed year/make/model/trim table in first 6000 chars |
| C | descriptionHtml | Add buyer-intent use-case language |
| D | tags | Add vehicle-specific tags (year ranges, generations, trims) |
| E | productType | Set Shopify standard taxonomy category |
| F | metafields | CONTROL: fitment only in metafields, removed from text |

## Shopify Catalog MCP

The experiment runs against the [Shopify Global Catalog](https://shopify.dev/docs/agents/catalog), the same index that powers AI shopping agents across ChatGPT, Gemini, and the Shop assistant. It exposes three MCP tools over a single HTTP endpoint.

**Endpoint**

```
Global:     https://catalog.shopify.com/api/ucp/mcp
Storefront: https://{storeDomain}/api/ucp/mcp
```

Requests include an agent profile URL in `meta.ucp-agent.profile` for capability negotiation. Keyless access works at low volume; authenticated keys (via Dev Dashboard) unlock higher rate limits.

**search_catalog**

Finds products using natural language, images, or product IDs.

```json
{
  "catalog.query": "bed organizer for 2023 Toyota Tacoma 3rd gen",
  "catalog.context": {
    "country": "US",
    "language": "en",
    "currency": "USD",
    "intent": "purchase"
  },
  "catalog.filters": {
    "available_for_sale": true,
    "ships_to_country": "US"
  },
  "catalog.pagination.limit": 10
}
```

Results are clustered by Universal Product ID (UPID) across all merchants. This is the tool used in every eval cycle to measure where each product ranks for a given query.

**lookup_catalog**

Retrieves up to 50 products by GID. Used by `watch_index.py` to verify a specific product's indexed field values after a mutation.

```json
{
  "catalog.ids": ["gid://shopify/Product/123456789"],
  "catalog.context": { "country": "US" }
}
```

**get_product**

Returns full product detail with variant availability and checkout links. Accepts option selections with relaxed matching so agents can handle partial selections gracefully.

```json
{
  "catalog.id": "gid://shopify/Product/123456789",
  "catalog.selected": [{ "name": "Size", "label": "6 ft" }],
  "catalog.preferences": ["Size", "Color"]
}
```

**Product schema**

```
id             UPID (gid://shopify/p/... format, stable across merchants)
title          Product title
description    HTML or plain text
url            Product page URL
media[]        Images with alt text (must render real-time, not cached)
price_range    { min, max, currency_code } in minor units
variants[]     Per-variant: price, availability, sku, selected_options, checkout_url
options[]      { name, values[] } with available/exists signals per value
availability   { in_stock, running_low }
seller         { shop_name, shop_id, shop_domain, shop_url }
tags[]         Merchant-defined tags
condition      new | secondhand
rating         { value, scale_max, count }
taxonomy[]     { id, name, type: "shopify_standard" | "merchant" }
```

Key nuance for this experiment: several fields (notably taxonomy and some availability signals) are **AI-inferred** by Shopify from the product's text content. This is why fitment data in structured metafields does not reliably influence ranking. The inference pipeline reads title and description, not custom namespaces.

## Stack

Python 3.9+. httpx for the Catalog MCP and Admin GraphQL. Anthropic SDK for mutation proposals and prompt generation. YAML prompt sets. File-based state. System cron for the index watcher.

The Catalog MCP client speaks the [MCP Streamable HTTP transport](https://spec.modelcontextprotocol.io) directly over httpx. No SDK wrapper needed.

No external database. Each component is a standalone script.

## Key Files

```
scrape.py          Ingest from Shopify Admin API or CSV/JSON export
prompts_gen.py     Generate vehicle-specific search queries via Claude
catalog_client.py  Shopify Catalog MCP client (search, lookup, get_product)
admin_client.py    Shopify Admin GraphQL (productUpdate)
mutate.py          Propose and apply one field change per cycle
watch_index.py     Cron: detect re-index, trigger evaluation and decision
eval.py            Run searches, record rank across query tiers
decide.py          Keep or revert based on rank change vs baseline
run_baseline.py    Capture the before state, output evals/baseline-report.md
pipeline.py        Orchestrator (start, run, status, install-cron)
dashboard.py       Terminal + HTML timeline of changes and ranking scores
```

## Setup

```bash
cp .env.example .env
# Fill in: SHOPIFY_SHOP_DOMAIN, SHOPIFY_ADMIN_TOKEN, ANTHROPIC_API_KEY

pip install -r requirements.txt

# Generate queries + capture baseline ranking
python pipeline.py start

# Apply first field change to all products
python pipeline.py run

# View progress
python dashboard.py
python dashboard.py --field-impact   # ranked table of which fields moved ranking
```

Requires a Shopify Agentic-plan store. Use a separate eval store, not production.

## Deliverables

After running all products through classes A-F:

- **Field-impact table**: which fields moved ranking and by how much
- **Per-product change summary**: the kept mutations per product, ready to apply in a PIM or content workflow
