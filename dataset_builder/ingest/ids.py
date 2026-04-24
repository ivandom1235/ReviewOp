from __future__ import annotations

import hashlib
from typing import Any


def _stable_hash(parts: list[str]) -> str:
    normalized = "|".join(part.strip().lower() for part in parts)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def stable_review_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("review_id") or row.get("id") or "").strip()
    if explicit:
        return explicit
    return "r_" + _stable_hash([
        str(row.get("source") or row.get("source_name") or "unknown"),
        str(row.get("text") or row.get("review_text") or ""),
    ])


def stable_group_id(row: dict[str, Any]) -> str:
    for key in ("group_id", "product_id", "business_id", "entity_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError("cannot create stable group_id without group_id/product_id/business_id/entity_id")
