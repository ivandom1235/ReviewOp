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

    # Calculate temporal sort key per group (fallback to synthetic check)
    def get_group_time(bucket):
        times = [r.get("timestamp") or r.get("date") for r in bucket if r.get("timestamp") or r.get("date")]
        return min(times) if times else "9999-99-99"

    synthetic_group_ids = set()
    for rid, bucket in grouped.items():
        if any(r.get("is_synthetic") or r.get("is_augmented") or r.get("source_type") == "augmented" for r in bucket):
            synthetic_group_ids.add(rid)

    ids = list(grouped.keys())
    rng = random.Random(seed)
    
    # Sort groups temporally first, tie-break with random shuffle
    ids_with_time = [(rid, get_group_time(grouped[rid]), rng.random()) for rid in ids]
    ids_with_time.sort(key=lambda x: (x[1], x[2]))
    sorted_ids = [x[0] for x in ids_with_time]

    # Force synthetic groups entirely to 'train' to prevent leakage
    train_ids = set(synthetic_group_ids)
    remaining_ids = [rid for rid in sorted_ids if rid not in train_ids]

    n = len(ids)
    n_train_target = max(1, int(n * ratios.get("train", 0.8)))
    n_val_target = max(1, int(n * ratios.get("val", 0.1))) if n >= 3 else max(0, n - n_train_target)

    # Fill remaining capacity for train from sorted temporal records
    needed_for_train = max(0, n_train_target - len(train_ids))
    train_ids.update(remaining_ids[:needed_for_train])
    remaining_ids = remaining_ids[needed_for_train:]

    val_ids = set(remaining_ids[:n_val_target])

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

def apply_domain_mixing(rows: List[Dict], max_open_share: float = 0.4, gold_eval_only: bool = True) -> List[Dict]:
    from collections import Counter
    import random
    
    if gold_eval_only:
        for r in rows:
            source = str(r.get("source", "")).lower()
            if "semeval" in source or "mams" in source:
                r["split"] = "test"
                
    train_rows = [r for r in rows if r.get("split") == "train"]
    domain_counts = Counter(str(r.get("domain", "")).lower() for r in train_rows)
    target_domain = domain_counts.most_common(1)[0][0] if domain_counts else ""
        
    for r in rows:
        source = str(r.get("source", "")).lower()
        domain = str(r.get("domain", "")).lower()
        if "semeval" in source or "mams" in source:
            r["source_type"] = "Type_C"
            r["sample_weight"] = 1.0
        elif domain == target_domain:
            r["source_type"] = "Type_A"
            r["sample_weight"] = 2.0
        else:
            r["source_type"] = "Type_B"
            r["sample_weight"] = 1.0
            
    type_a = [r for r in train_rows if r.get("source_type") == "Type_A"]
    type_b = [r for r in train_rows if r.get("source_type") == "Type_B"]
    other_rows = [r for r in rows if r.get("split") != "train"]
    
    max_b = int(len(type_a) * max_open_share / (1.0 - max_open_share)) if type_a else len(type_b)
    if len(type_b) > max_b:
        rng = random.Random(42)
        rng.shuffle(type_b)
        type_b = type_b[:max_b]
        
    return type_a + type_b + other_rows
