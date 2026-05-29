from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _stable_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, default=str)
    return "yelp_" + hashlib.md5(payload.encode()).hexdigest()[:12]


def _primary_category(categories: Any) -> str:
    """
    Extract a normalized primary category from Yelp categories field.
    Yelp categories can be a comma-separated string or a list.
    Falls back to "general" when unavailable.
    """
    if not categories:
        return "general"
    if isinstance(categories, list):
        categories = categories[0] if categories else "general"
    first = str(categories).split(",")[0].strip()
    if not first:
        return "general"
    return first.lower().replace(" ", "_").replace("&", "and")


def adapt_yelp_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes Yelp Academic Dataset rows to canonical schema.
    Domain is derived from Yelp categories — never hardcoded to "restaurant".
    """
    review_id = (
        str(row.get("review_id") or "").strip()
        or _stable_hash(row)
    )

    product_id = str(row.get("business_id") or "").strip() or None

    raw_categories = row.get("categories") or row.get("business_categories")
    domain = "yelp:" + _primary_category(raw_categories)

    text = str(row.get("text") or "").strip()

    rating = row.get("stars") or row.get("rating")
    if rating is not None:
        try:
            rating = float(rating)
        except (ValueError, TypeError):
            rating = None

    if isinstance(raw_categories, str):
        categories_list = [c.strip() for c in raw_categories.split(",") if c.strip()]
    elif isinstance(raw_categories, list):
        categories_list = raw_categories
    else:
        categories_list = []

    return {
        "review_id": review_id,
        "product_id": product_id,
        "domain": domain,
        "text": text,
        "rating": rating,
        "source_dataset": "yelp",
        "metadata": {
            "business_id": row.get("business_id"),
            "categories": categories_list,
            "date": row.get("date"),
            "useful": row.get("useful"),
            "funny": row.get("funny"),
            "cool": row.get("cool"),
        },
    }
