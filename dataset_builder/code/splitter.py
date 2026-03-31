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
