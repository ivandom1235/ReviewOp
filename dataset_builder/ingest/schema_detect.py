from __future__ import annotations

from typing import Any


TEXT_FIELDS = ("review_text", "text", "review", "content", "body", "comment")


def infer_text_field(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        if str(row.get(field) or "").strip():
            return field
    raise ValueError("no text field found")


def detect_dataset_schema(row: dict[str, Any]) -> dict[str, str]:
    return {
        "text_field": infer_text_field(row),
        "domain_field": "domain" if "domain" in row else "",
        "group_field": next((key for key in ("group_id", "product_id", "business_id", "entity_id") if key in row), ""),
    }
