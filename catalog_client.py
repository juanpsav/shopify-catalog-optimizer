"""
catalog_client.py — Shopify Global Catalog MCP client.

Connects to https://catalog.shopify.com/api/ucp/mcp (or a storefront-scoped
endpoint) and exposes three operations:

  search_catalog(query, limit, country)  → ranked list of products
  lookup_catalog(shop_domain, product_id) → a specific merchant's product
  get_product(upid)                       → full product detail by UPID

Authentication:
  - Keyless: no credentials needed for low-volume access
  - Authenticated: set CATALOG_API_KEY env var for higher rate limits

All calls are async. Use asyncio.run() or the async helpers at the bottom for
synchronous callers (eval.py, watch_index.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)


# ─── Response types ───────────────────────────────────────────────────────────

class CatalogProduct:
    """Lightweight wrapper around a single search result."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        # Normalize common field paths from MCP tool response
        self.upid: str = raw.get("upid", raw.get("id", ""))
        self.title: str = raw.get("title", "")
        self.vendor: str = raw.get("vendor", raw.get("brand", ""))
        self.price: str = raw.get("price", raw.get("priceRange", ""))
        self.shop_domain: str = raw.get("shopDomain", raw.get("shop_domain", ""))
        self.description: str = raw.get("description", raw.get("body", ""))
        self.tags: list[str] = raw.get("tags", [])
        self.product_type: str = raw.get("productType", raw.get("product_type", ""))

    def __repr__(self) -> str:
        return f"<CatalogProduct upid={self.upid!r} title={self.title!r}>"


class SearchResult:
    """A ranked list of products returned by search_catalog."""

    def __init__(self, query: str, products: list[CatalogProduct]):
        self.query = query
        self.products = products

    def rank_of(self, product_id: str, shop_domain: str | None = None) -> int | None:
        """Return the 1-based rank of a product, or None if not found."""
        for i, p in enumerate(self.products, start=1):
            if product_id and (
                p.upid == product_id
                or p.upid.endswith(product_id)
                or product_id.endswith(p.upid)
            ):
                if shop_domain is None or p.shop_domain == shop_domain:
                    return i
            if shop_domain and p.shop_domain == shop_domain and not product_id:
                return i
        return None

    def reciprocal_rank(self, product_id: str, shop_domain: str | None = None) -> float:
        """Return MRR contribution: 1/rank, or 0.0 if not found."""
        rank = self.rank_of(product_id, shop_domain)
        return 1.0 / rank if rank is not None else 0.0


# ─── MCP client (async) ───────────────────────────────────────────────────────

class CatalogMCPClient:
    """
    Thin async client for the Shopify Catalog MCP endpoint.

    We call the MCP endpoint using the Streamable HTTP transport as described in
    the MCP specification. For each tool call we:
      1. POST an initialize request (session handshake)
      2. POST a tools/call request
      3. Parse the streamed NDJSON response

    This is a direct HTTP implementation that avoids mcp-sdk version pinning issues
    while following the MCP protocol exactly.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.url = url or config.catalog_mcp_url()
        self.api_key = api_key if api_key is not None else config.CATALOG_API_KEY
        self.timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a single MCP tool call via the Streamable HTTP transport.

        The MCP Streamable HTTP spec requires:
          - POST to <endpoint>  (not /sse, not /messages)
          - Body: JSON-RPC 2.0 with method = "tools/call"
          - The server may respond with a single JSON object or NDJSON stream

        We send initialize + tools/call in sequence on the same session.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Step 1: Initialize session
            init_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "catalog-optimizer", "version": "1.0.0"},
                },
            }
            init_resp = await client.post(
                self.url, json=init_payload, headers=self._headers()
            )
            init_resp.raise_for_status()

            # Parse session ID from response headers (Mcp-Session-Id)
            session_id = init_resp.headers.get("Mcp-Session-Id", "")
            call_headers = self._headers()
            if session_id:
                call_headers["Mcp-Session-Id"] = session_id

            # Step 2: Call the tool
            call_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            call_resp = await client.post(
                self.url, json=call_payload, headers=call_headers
            )
            call_resp.raise_for_status()

            # Parse response — may be JSON or NDJSON stream
            content_type = call_resp.headers.get("content-type", "")
            if "text/event-stream" in content_type or "ndjson" in content_type:
                return self._parse_sse_response(call_resp.text)
            else:
                return call_resp.json()

    def _parse_sse_response(self, text: str) -> dict[str, Any]:
        """Parse SSE/NDJSON response and extract the last meaningful JSON object."""
        result: dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    try:
                        obj = json.loads(data)
                        if "result" in obj:
                            result = obj
                    except json.JSONDecodeError:
                        pass
            elif line and not line.startswith(":"):
                try:
                    obj = json.loads(line)
                    if "result" in obj:
                        result = obj
                except json.JSONDecodeError:
                    pass
        return result

    def _extract_content(self, response: dict[str, Any]) -> Any:
        """Extract the actual content from an MCP tool response."""
        # JSON-RPC success: {"jsonrpc": "2.0", "id": ..., "result": {...}}
        result = response.get("result", response)

        # MCP tool result: {"content": [{"type": "text", "text": "..."}]}
        content_list = result.get("content", [])
        if content_list and isinstance(content_list, list):
            first = content_list[0]
            if isinstance(first, dict) and first.get("type") == "text":
                text = first.get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text

        # Direct result payload
        return result

    async def search_catalog(
        self,
        query: str,
        limit: int = 10,
        country: str | None = None,
        available_for_sale: bool = True,
    ) -> SearchResult:
        """
        Search the Global Catalog for products matching `query`.

        Returns a SearchResult with up to `limit` ranked CatalogProduct objects.
        """
        args: dict[str, Any] = {
            "query": query,
            "first": limit,
            "availableForSale": available_for_sale,
        }
        if country:
            args["country"] = country

        try:
            raw = await self._call_tool("search_catalog", args)
            data = self._extract_content(raw)
        except Exception as exc:
            logger.warning("search_catalog failed for %r: %s", query, exc)
            return SearchResult(query=query, products=[])

        products = self._parse_search_results(data)
        logger.debug("search_catalog(%r) → %d results", query, len(products))
        return SearchResult(query=query, products=products)

    def _parse_search_results(self, data: Any) -> list[CatalogProduct]:
        """Normalize various response shapes into a list of CatalogProduct."""
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            for key in ("products", "results", "items", "edges", "nodes"):
                if key in data:
                    items = data[key]
                    # GraphQL edges/nodes unwrap
                    if items and isinstance(items[0], dict) and "node" in items[0]:
                        items = [e["node"] for e in items]
                    break
            else:
                # Treat the dict itself as a single result
                items = [data]
        else:
            items = []

        return [CatalogProduct(item) for item in items if isinstance(item, dict)]

    async def lookup_catalog(
        self, shop_domain: str, product_id: str
    ) -> CatalogProduct | None:
        """
        Look up a specific merchant product in the catalog.

        Use this to verify that the latest field values are indexed.
        `product_id` should be the Shopify GID (e.g. gid://shopify/Product/123)
        or just the numeric ID.
        """
        args = {"shopDomain": shop_domain, "productId": product_id}
        try:
            raw = await self._call_tool("lookup_catalog", args)
            data = self._extract_content(raw)
        except Exception as exc:
            logger.warning(
                "lookup_catalog(%r, %r) failed: %s", shop_domain, product_id, exc
            )
            return None

        if isinstance(data, dict) and data:
            return CatalogProduct(data)
        if isinstance(data, list) and data:
            return CatalogProduct(data[0])
        return None

    async def get_product(self, upid: str) -> CatalogProduct | None:
        """
        Retrieve detailed product info by Universal Product ID (UPID).

        Use this after lookup_catalog to verify indexed field values,
        or to confirm a re-index has occurred.
        """
        args = {"upid": upid}
        try:
            raw = await self._call_tool("get_product", args)
            data = self._extract_content(raw)
        except Exception as exc:
            logger.warning("get_product(%r) failed: %s", upid, exc)
            return None

        if isinstance(data, dict) and data:
            return CatalogProduct(data)
        return None


# ─── Sync helpers (for use in non-async callers) ─────────────────────────────

def search_sync(
    query: str, limit: int = 10, country: str | None = None
) -> SearchResult:
    client = CatalogMCPClient()
    return asyncio.run(client.search_catalog(query, limit=limit, country=country))


def lookup_sync(shop_domain: str, product_id: str) -> CatalogProduct | None:
    client = CatalogMCPClient()
    return asyncio.run(client.lookup_catalog(shop_domain, product_id))


def get_product_sync(upid: str) -> CatalogProduct | None:
    client = CatalogMCPClient()
    return asyncio.run(client.get_product(upid))


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "truck bed organizer"
    print(f"Searching catalog for: {query!r}")
    result = search_sync(query, limit=5)
    if not result.products:
        print("No results (or catalog MCP not reachable in this environment)")
    for i, p in enumerate(result.products, 1):
        print(f"  {i}. [{p.upid}] {p.title!r} — {p.shop_domain}")
