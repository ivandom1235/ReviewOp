from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _stable_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return "csv_" + hashlib.md5(payload.encode()).hexdigest()[:12]


def adapt_generic_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a generic CSV/JSONL review row to canonical schema.
    Accepts the broadest possible field set for maximum compatibility.
    """
    review_id = (
        str(row.get("review_id") or row.get("id") or "").strip()
        or _stable_hash(row)
    )

    product_id = (
        str(row.get("product_id") or row.get("asin") or row.get("business_id") or "").strip()
        or None
    )

    domain = str(row.get("domain") or row.get("category") or "generic").lower().strip()

    text = str(
        row.get("text")
        or row.get("review_body")
        or row.get("review_text")
        or row.get("content")
        or ""
    ).strip()

    rating = row.get("rating") or row.get("stars") or row.get("overall") or row.get("score")
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    known_keys = {
        "review_id", "id", "product_id", "asin", "business_id",
        "domain", "category", "text", "review_body", "review_text",
        "content", "rating", "stars", "overall", "score",
    }
    metadata = {k: v for k, v in row.items() if k not in known_keys}

    return {
        "review_id": review_id,
        "product_id": product_id,
        "domain": domain,
        "text": text,
        "rating": rating,
        "source_dataset": "csv",
        "metadata": metadata,
    }
