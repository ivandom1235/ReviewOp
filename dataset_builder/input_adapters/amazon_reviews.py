from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _stable_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return "amazon_" + hashlib.md5(payload.encode()).hexdigest()[:12]


def _normalize_category(raw: str | None) -> str:
    if not raw:
        return "general"
    return raw.lower().strip().replace(" ", "_").replace("&", "and")


def adapt_amazon_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes Amazon review rows (TSV/JSON, old format or Amazon Reviews 2023)
    to the canonical schema. Supports both legacy and 2023 field names.
    """
    review_id = (
        str(row.get("review_id") or "").strip()
        or str(row.get("id") or "").strip()
        or _stable_hash(row)
    )

    product_id = (
        str(row.get("parent_asin") or "").strip()
        or str(row.get("asin") or "").strip()
        or str(row.get("product_id") or "").strip()
        or None
    )

    raw_category = (
        row.get("main_category")
        or row.get("product_category")
        or row.get("categories")
        or "general"
    )
    if isinstance(raw_category, list):
        raw_category = raw_category[0] if raw_category else "general"
    domain = "amazon:" + _normalize_category(str(raw_category))

    title = str(row.get("title") or row.get("review_headline") or "").strip()
    body = str(row.get("text") or row.get("review_body") or "").strip()
    text = f"{title}. {body}".strip(". ") if title else body

    rating = (
        row.get("rating")
        or row.get("overall")
        or row.get("star_rating")
    )
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    return {
        "review_id": review_id,
        "product_id": product_id,
        "domain": domain,
        "text": text,
        "rating": rating,
        "source_dataset": "amazon",
        "metadata": {
            "parent_asin": row.get("parent_asin") or row.get("asin"),
            "main_category": raw_category,
            "verified_purchase": row.get("verified_purchase"),
            "helpful_vote": row.get("helpful_vote") or row.get("helpful_votes"),
            "timestamp": row.get("timestamp") or row.get("date"),
        },
    }
