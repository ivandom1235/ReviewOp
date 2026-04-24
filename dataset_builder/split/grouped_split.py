from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def _group_id(row: Any) -> str:
    value = getattr(row, "group_id", None)
    if value is None and isinstance(row, dict):
        value = row.get("group_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("group_id is required for grouped split")
    return value


def grouped_train_val_test_split(
    rows: list[Any],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[Any]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    by_group: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_group[_group_id(row)].append(row)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    train_end = max(1, int(round(n * train_ratio))) if n else 0
    val_end = min(n, train_end + int(round(n * val_ratio)))
    split_groups = {
        "train": groups[:train_end],
        "val": groups[train_end:val_end],
        "test": groups[val_end:],
    }
    return {split: [row for group in group_ids for row in by_group[group]] for split, group_ids in split_groups.items()}
