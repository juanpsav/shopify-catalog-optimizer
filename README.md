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

## Stack

Python 3.9+. httpx for the Catalog MCP and Admin GraphQL. Anthropic SDK for mutation proposals and prompt generation. YAML prompt sets. File-based state. System cron for the index watcher.

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
