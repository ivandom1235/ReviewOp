from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_stratify(values: Sequence[str] | None) -> List[str] | None:
    if not values:
        return None
    series = pd.Series(list(values)).fillna("unknown").astype(str)
    counts = series.value_counts()
    if counts.empty or int(counts.min()) < 2:
        return None
    return series.tolist()


def choose_stratify_values(rows: Iterable[dict], *, preferred_key: str | None = None, fallback_key: str | None = None) -> tuple[str | None, List[str] | None]:
    materialized = list(rows)
    for key in (preferred_key, fallback_key):
        if not key:
            continue
        values = [str(row.get(key, "unknown")) for row in materialized]
        safe = _safe_stratify(values)
        if safe is not None:
            return key, safe
    return None, None


def preliminary_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    random_seed: int,
    stratify_values: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 2:
        return frame.reset_index(drop=True), frame.iloc[0:0].reset_index(drop=True)
    train_frame, holdout_frame = train_test_split(
        frame,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=_safe_stratify(stratify_values),
    )
    return train_frame.reset_index(drop=True), holdout_frame.reset_index(drop=True)


def split_holdout(
    holdout_rows: List[dict],
    *,
    val_ratio_within_holdout: float,
    random_seed: int,
    stratify_values: Sequence[str] | None = None,
) -> tuple[List[dict], List[dict]]:
    if not holdout_rows:
        return [], []
    if len(holdout_rows) == 1:
        row = holdout_rows[0]
        return ([row], []) if val_ratio_within_holdout >= 0.5 else ([], [row])
    indices = list(range(len(holdout_rows)))
    val_idx, test_idx = train_test_split(
        indices,
        train_size=val_ratio_within_holdout,
        random_state=random_seed,
        shuffle=True,
        stratify=_safe_stratify(stratify_values),
    )
    val_set = set(int(idx) for idx in val_idx)
    val_rows = [row for idx, row in enumerate(holdout_rows) if idx in val_set]
    test_rows = [row for idx, row in enumerate(holdout_rows) if idx not in val_set]
    return val_rows, test_rows


def grouped_split(
    rows: List[dict],
    *,
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[List[dict], List[dict], List[dict]]:
    if not rows:
        return [], [], []
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    by_group: dict[str, List[dict]] = {}
    for row in rows:
        key = str(row.get(group_key) or "unknown")
        by_group.setdefault(key, []).append(row)
    groups = list(by_group.keys())
    if len(groups) == 1:
        grp = groups[0]
        return by_group[grp], [], []

    train_groups, holdout_groups = train_test_split(
        groups,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True,
    )
    holdout_ratio = val_ratio / max(val_ratio + test_ratio, 1e-9)
    if not holdout_groups:
        holdout_groups = []
    
    if len(holdout_groups) == 0 and len(groups) > 1 and test_ratio > 0:
        # Emergency fallback: if we have groups but holdout is empty, 
        # steal one group from train for test if possible.
        if len(train_groups) > 1:
            test_groups = [train_groups.pop()]
            val_groups = [] # Still might be empty if only 2 groups total
        else:
            val_groups = []
            test_groups = []
    elif len(holdout_groups) == 1:
        # Priority: test > val for research benchmarks
        if test_ratio > 0:
            test_groups = holdout_groups
            val_groups = []
        else:
            val_groups = holdout_groups
            test_groups = []
    else:
        val_groups, test_groups = train_test_split(
            holdout_groups,
            test_size=test_ratio / max(val_ratio + test_ratio, 1e-9),
            random_state=random_seed + 1,
            shuffle=True,
        )
    train = [row for group in train_groups for row in by_group[group]]
    val = [row for group in val_groups for row in by_group[group]]
    test = [row for group in test_groups for row in by_group[group]]
    return train, val, test


def grouped_leakage_report(
    rows: List[dict],
    *,
    group_key: str,
    split_key: str = "split",
) -> dict:
    train_groups = {str(row.get(group_key) or "unknown") for row in rows if str(row.get(split_key, "train")) == "train"}
    val_groups = {str(row.get(group_key) or "unknown") for row in rows if str(row.get(split_key, "train")) == "val"}
    test_groups = {str(row.get(group_key) or "unknown") for row in rows if str(row.get(split_key, "train")) == "test"}
    overlap_train_val = sorted(train_groups & val_groups)
    overlap_train_test = sorted(train_groups & test_groups)
    overlap_val_test = sorted(val_groups & test_groups)
    return {
        "group_key": group_key,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "overlap_counts": {
            "train_val": len(overlap_train_val),
            "train_test": len(overlap_train_test),
            "val_test": len(overlap_val_test),
        },
        "overlap_samples": {
            "train_val": overlap_train_val[:20],
            "train_test": overlap_train_test[:20],
            "val_test": overlap_val_test[:20],
        },
    }
