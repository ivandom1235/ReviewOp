from __future__ import annotations

from typing import Any


def domain_holdout_split(rows: list[Any], holdout_domain: str) -> dict[str, list[Any]]:
    train, test = [], []
    for row in rows:
        domain = str(getattr(row, "domain", row.get("domain") if isinstance(row, dict) else ""))
        (test if domain == holdout_domain else train).append(row)
    return {"train": train, "val": [], "test": test}
