from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_domain(domain: str | None) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(domain or "unknown").lower()).strip("_")
    return value or "unknown"


def normalize_metadata(metadata: dict | None) -> dict:
    return dict(metadata or {})
