from __future__ import annotations

from collections import Counter
from dataclasses import is_dataclass, replace
from typing import Any


def _row_domain(row: Any) -> str:
    value = getattr(row, "domain", None)
    if value is None and isinstance(row, dict):
        value = row.get("domain")
    return str(value or "").strip()


def choose_domain_holdout_domain(rows: list[Any]) -> str:
    counts = Counter(domain for domain in (_row_domain(row) for row in rows) if domain and domain.lower() != "unknown")
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def domain_holdout_split(rows: list[Any], holdout_domain: str | None = None) -> dict[str, list[Any]]:
    chosen_domain = str(holdout_domain or choose_domain_holdout_domain(rows) or "unknown").strip()
    train_like: list[Any] = []
    test: list[Any] = []
    for row in rows:
        domain = _row_domain(row)
        if domain and domain == chosen_domain:
            test.append(row)
        else:
            train_like.append(row)

    n = len(train_like)
    if n >= 50:
        val_target = max(10, round(n * 0.10))
        if n >= 300:
            val_target = max(val_target, 50)
    elif n >= 10:
        val_target = max(5, round(n * 0.20))
    else:
        val_target = max(1, round(n * 0.20))

    import random
    random.Random(42).shuffle(train_like)
    val = train_like[:val_target]
    train = train_like[val_target:]

    out = {"train": train, "val": val, "test": test}
    for split, split_rows in out.items():
        for idx, row in enumerate(split_rows):
            existing = getattr(row, "split_protocol", None)
            if existing is None and isinstance(row, dict):
                existing = row.get("split_protocol")
            proto = dict(existing) if isinstance(existing, dict) else {}
            proto.update({"random": "unused", "grouped": "unused", "domain_holdout": split})
            if is_dataclass(row):
                split_rows[idx] = replace(row, split_protocol=proto)
            elif isinstance(row, dict):
                row["split_protocol"] = proto
    return out
