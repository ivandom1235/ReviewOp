"""Split logic with review-level grouping and leakage checks."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def assign_splits(rows: List[Dict], ratios: Dict[str, float], seed: int = 42) -> List[Dict]:
    if not rows:
        return rows

    has_declared = any(str(r.get("split", "")).strip().lower() in {"train", "val", "test"} for r in rows)
    if has_declared:
        for r in rows:
            split = str(r.get("split", "")).strip().lower()
            r["split"] = split if split in {"train", "val", "test"} else "train"
        return rows

    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)

    ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, int(n * ratios.get("train", 0.8)))
    n_val = max(1, int(n * ratios.get("val", 0.1))) if n >= 3 else max(0, n - n_train)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])

    for rid, bucket in grouped.items():
        split = "train" if rid in train_ids else "val" if rid in val_ids else "test"
        for row in bucket:
            row["split"] = split

    return rows


def split_rows(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split", "train")).lower()
        if split not in out:
            split = "train"
        out[split].append(row)
    return out


def leakage_ids(split_map: Dict[str, List[Dict]]) -> List[Tuple[str, str, str]]:
    ids_by_split = {k: {str(r.get("id")) for r in v} for k, v in split_map.items()}
    overlaps = []
    keys = ["train", "val", "test"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = ids_by_split[a].intersection(ids_by_split[b])
            for cid in sorted(common):
                overlaps.append((a, b, cid))
    return overlaps
