"""
admin_client.py — Shopify Admin GraphQL client.

Provides two operations:
  get_product(product_id)               → current field values from Admin API
  update_product(product_id, fields)    → apply a mutation via productUpdate

Authentication: SHOPIFY_SHOP_DOMAIN + SHOPIFY_ADMIN_TOKEN env vars.
Rate limiting: leaky-bucket, check throttleStatus in response extensions.

Usage:
    client = AdminClient()
    product = client.get_product("gid://shopify/Product/123456789")
    updated = client.update_product("gid://shopify/Product/123456789", {
        "title": "New Title with Fitment",
        "tags": ["tag1", "tag2"],
    })
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)


# ─── GraphQL fragments ────────────────────────────────────────────────────────

PRODUCT_FIELDS_FRAGMENT = """
fragment ProductFields on Product {
  id
  title
  descriptionHtml
  tags
  productType
  vendor
  status
  handle
  updatedAt
  seo {
    title
    description
  }
  metafields(first: 20) {
    edges {
      node {
        id
        namespace
        key
        type
        value
        updatedAt
      }
    }
  }
  category {
    id
    name
    fullName
  }
}
"""

GET_PRODUCT_QUERY = (
    PRODUCT_FIELDS_FRAGMENT
    + """
query GetProduct($id: ID!) {
  product(id: $id) {
    ...ProductFields
  }
}
"""
)

UPDATE_PRODUCT_MUTATION = """
mutation UpdateProduct($id: ID!, $input: ProductInput!) {
  productUpdate(product: $input) {
    product {
      id
      title
      descriptionHtml
      tags
      productType
      updatedAt
      metafields(first: 20) {
        edges {
          node {
            id
            namespace
            key
            type
            value
            updatedAt
          }
        }
      }
    }
    userErrors {
      field
      message
    }
  }
}
"""

# ─── Throttle handling ────────────────────────────────────────────────────────

class ThrottleError(Exception):
    """Raised when the Admin API bucket is exhausted."""


class AdminAPIError(Exception):
    """Raised when Admin API returns user errors or HTTP errors."""


# ─── Client ───────────────────────────────────────────────────────────────────

class AdminClient:
    """
    Synchronous Shopify Admin GraphQL client.

    Thin wrapper around httpx — every call is a single POST with
    automatic throttle detection and exponential back-off.
    """

    def __init__(self, max_retries: int = 3, initial_backoff: float = 2.0):
        self._url = config.shopify_graphql_url()
        self._headers = config.shopify_headers()
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def _execute(
        self, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a GraphQL request with retry on throttle."""
        backoff = self.initial_backoff
        last_error: Exception = RuntimeError("no attempts made")

        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    self._url,
                    json={"query": query, "variables": variables},
                    headers=self._headers,
                    timeout=30.0,
                )
                resp.raise_for_status()
                body = resp.json()

                # Check cost extensions for throttle status
                extensions = body.get("extensions", {})
                throttle = extensions.get("cost", {}).get("throttleStatus", {})
                if throttle.get("currentlyAvailable", 1) < 50:
                    restore_rate = throttle.get("restoreRate", 50)
                    wait = max(1.0, 50 / max(restore_rate, 1))
                    logger.debug("Throttle low — waiting %.1fs", wait)
                    time.sleep(wait)

                errors = body.get("errors", [])
                if errors:
                    msgs = "; ".join(e.get("message", str(e)) for e in errors)
                    raise AdminAPIError(f"GraphQL errors: {msgs}")

                return body.get("data", {})

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning(
                        "429 from Admin API (attempt %d/%d), backing off %.1fs",
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    last_error = exc
                    continue
                raise

        raise last_error

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_product(self, product_id: str) -> dict[str, Any]:
        """
        Fetch current product fields from Shopify Admin API.

        Returns a dict matching the ProductFields fragment above.
        Flattens metafields edges/nodes for convenience.
        """
        data = self._execute(GET_PRODUCT_QUERY, {"id": product_id})
        product = data.get("product")
        if product is None:
            raise AdminAPIError(f"Product not found: {product_id!r}")
        return self._normalize_product(product)

    def update_product(
        self,
        product_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply a mutation to a product via productUpdate.

        `fields` maps Shopify input field names to new values, e.g.:
          {
            "title": "New Title",
            "tags": ["tag1", "tag2"],
            "descriptionHtml": "<p>...</p>",
            "metafields": [{"namespace": "fitment", "key": "vehicles",
                            "type": "json", "value": "..."}],
          }

        Returns the updated product normalized dict.
        Raises AdminAPIError if userErrors is non-empty.
        """
        # productUpdate requires 'id' inside the input
        input_payload: dict[str, Any] = {"id": product_id, **fields}

        data = self._execute(
            UPDATE_PRODUCT_MUTATION,
            {"id": product_id, "input": input_payload},
        )

        result = data.get("productUpdate", {})
        user_errors = result.get("userErrors", [])
        if user_errors:
            msgs = "; ".join(
                f"{e.get('field', '?')}: {e.get('message', str(e))}"
                for e in user_errors
            )
            raise AdminAPIError(f"productUpdate userErrors: {msgs}")

        product = result.get("product")
        if product is None:
            raise AdminAPIError("productUpdate returned no product")
        return self._normalize_product(product)

    def restore_field(
        self,
        product_id: str,
        field_name: str,
        prior_value: Any,
    ) -> dict[str, Any]:
        """
        Convenience wrapper: restore a single field to its prior value.
        Used by decide.py on REVERT.
        """
        return self.update_product(product_id, {field_name: prior_value})

    # ── Normalization ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_product(raw: dict[str, Any]) -> dict[str, Any]:
        """Flatten metafields edges/nodes, normalize category."""
        product = dict(raw)

        # Flatten metafields GraphQL edges
        mf_connection = product.pop("metafields", {})
        edges = (
            mf_connection.get("edges", []) if isinstance(mf_connection, dict) else []
        )
        product["metafields"] = [
            edge["node"] for edge in edges if "node" in edge
        ]

        # Flatten category
        cat = product.get("category")
        if isinstance(cat, dict):
            product["taxonomy_category"] = cat.get("fullName") or cat.get("name")
        else:
            product["taxonomy_category"] = None

        return product


# ─── Dry-run mode (no credentials needed for testing) ────────────────────────

class DryRunAdminClient(AdminClient):
    """
    Drop-in replacement for AdminClient that logs mutations instead of
    sending them. Useful during development and CI.
    """

    def __init__(self):
        # Skip credential loading
        self._url = "https://dry-run.local/"
        self._headers = {}
        self.max_retries = 1
        self.initial_backoff = 0.0

    def get_product(self, product_id: str) -> dict[str, Any]:
        logger.info("[DRY-RUN] get_product(%s)", product_id)
        return {
            "id": product_id,
            "title": "Dry-run product",
            "descriptionHtml": "",
            "tags": [],
            "productType": "",
            "vendor": "",
            "status": "ACTIVE",
            "handle": "dry-run",
            "updatedAt": "2025-01-01T00:00:00Z",
            "metafields": [],
            "taxonomy_category": None,
        }

    def update_product(self, product_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        logger.info(
            "[DRY-RUN] update_product(%s) fields=%s",
            product_id,
            json.dumps(fields, indent=2),
        )
        base = self.get_product(product_id)
        base.update(fields)
        return base


def get_admin_client(dry_run: bool = False) -> AdminClient:
    """Factory: returns a real or dry-run client based on env / flag."""
    if dry_run or not config.SHOPIFY_ADMIN_TOKEN:
        logger.info("Using DryRunAdminClient (no credentials configured)")
        return DryRunAdminClient()
    return AdminClient()


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    product_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not product_id:
        print("Usage: python admin_client.py <product_gid>")
        sys.exit(1)

    client = get_admin_client()
    product = client.get_product(product_id)
    print(json.dumps(product, indent=2))
