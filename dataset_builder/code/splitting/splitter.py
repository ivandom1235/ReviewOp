from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable, List, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from row_contracts import DedupChecked, SplitAssigned


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
    by_group: dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key) or "unknown")
        by_group[key].append(row)
    groups = list(by_group.keys())
    if len(groups) == 1:
        grp = groups[0]
        return by_group[grp], [], []

    total_rows = max(1, len(rows))
    target_rows = {
        "train": train_ratio * total_rows,
        "val": val_ratio * total_rows,
        "test": test_ratio * total_rows,
    }

    def _row_domain_family(row: dict) -> str:
        domain = str(row.get("domain_family") or row.get("domain") or "unknown").strip().lower()
        return domain or "unknown"

    def _row_labels(row: dict) -> list[str]:
        labels: list[str] = []
        interpretations = row.get("gold_interpretations")
        if isinstance(interpretations, list) and interpretations:
            for item in interpretations:
                if isinstance(item, dict):
                    aspect = str(item.get("aspect_label") or item.get("aspect") or "").strip().lower()
                    if aspect:
                        labels.append(aspect)
        if not labels:
            implicit = row.get("implicit", {}) if isinstance(row.get("implicit"), dict) else {}
            aspects = implicit.get("aspects") if isinstance(implicit.get("aspects"), list) else []
            labels = [str(a).strip().lower() for a in aspects if str(a).strip() and str(a).strip().lower() != "general"]
        return labels

    group_stats: dict[str, dict[str, object]] = {}
    total_label_counter: Counter[str] = Counter()
    total_domain_counter: Counter[str] = Counter()
    total_hardness_counter: Counter[str] = Counter()
    total_multi_label = 0
    total_abstain = 0
    total_novel = 0

    for group in groups:
        group_rows = by_group[group]
        label_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        hardness_counter: Counter[str] = Counter()
        multi_label_rows = 0
        abstain_rows = 0
        novel_rows = 0
        for row in group_rows:
            domain = _row_domain_family(row)
            domain_counter[domain] += 1
            total_domain_counter[domain] += 1
            labels = _row_labels(row)
            if len(set(labels)) >= 2:
                multi_label_rows += 1
                total_multi_label += 1
            for label in labels:
                label_counter[label] += 1
                total_label_counter[label] += 1
            hardness = str(
                (row.get("implicit", {}) or {}).get("hardness_tier")
                or row.get("hardness_tier")
                or "H0"
            ).strip().upper()
            hardness_counter[hardness] += 1
            total_hardness_counter[hardness] += 1
            if bool(row.get("abstain_acceptable", False)):
                abstain_rows += 1
                total_abstain += 1
            if bool(row.get("novel_acceptable", False)):
                novel_rows += 1
                total_novel += 1
        dominant_domain = max(domain_counter.items(), key=lambda item: (item[1], item[0]))[0] if domain_counter else "unknown"
        group_stats[group] = {
            "rows": len(group_rows),
            "dominant_domain": dominant_domain,
            "label_counter": label_counter,
            "domain_counter": domain_counter,
            "hardness_counter": hardness_counter,
            "multi_label_rows": multi_label_rows,
            "abstain_rows": abstain_rows,
            "novel_rows": novel_rows,
        }

    targets = {
        "label_counter": {k: train_ratio * v for k, v in total_label_counter.items()},
        "domain_counter": {k: train_ratio * v for k, v in total_domain_counter.items()},
        "hardness_counter": {k: train_ratio * v for k, v in total_hardness_counter.items()},
        "multi_label_rows": train_ratio * total_multi_label,
        "abstain_rows": train_ratio * total_abstain,
        "novel_rows": train_ratio * total_novel,
    }
    split_targets = {
        "train": targets,
        "val": {
            "label_counter": {k: val_ratio * v for k, v in total_label_counter.items()},
            "domain_counter": {k: val_ratio * v for k, v in total_domain_counter.items()},
            "hardness_counter": {k: val_ratio * v for k, v in total_hardness_counter.items()},
            "multi_label_rows": val_ratio * total_multi_label,
            "abstain_rows": val_ratio * total_abstain,
            "novel_rows": val_ratio * total_novel,
        },
        "test": {
            "label_counter": {k: test_ratio * v for k, v in total_label_counter.items()},
            "domain_counter": {k: test_ratio * v for k, v in total_domain_counter.items()},
            "hardness_counter": {k: test_ratio * v for k, v in total_hardness_counter.items()},
            "multi_label_rows": test_ratio * total_multi_label,
            "abstain_rows": test_ratio * total_abstain,
            "novel_rows": test_ratio * total_novel,
        },
    }

    split_group_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_stats = {
        "train": {"rows": 0, "label_counter": Counter(), "domain_counter": Counter(), "hardness_counter": Counter(), "multi_label_rows": 0, "abstain_rows": 0, "novel_rows": 0},
        "val": {"rows": 0, "label_counter": Counter(), "domain_counter": Counter(), "hardness_counter": Counter(), "multi_label_rows": 0, "abstain_rows": 0, "novel_rows": 0},
        "test": {"rows": 0, "label_counter": Counter(), "domain_counter": Counter(), "hardness_counter": Counter(), "multi_label_rows": 0, "abstain_rows": 0, "novel_rows": 0},
    }

    def _score_assignment(split: str, group_key_value: str) -> float:
        stats = split_stats[split]
        group_meta = group_stats[group_key_value]
        projected_rows = float(stats["rows"]) + float(group_meta["rows"])
        row_target = target_rows[split]
        row_penalty = abs(projected_rows - row_target) / max(1.0, row_target if row_target > 0 else 1.0)

        target_meta = split_targets[split]
        label_penalty = 0.0
        for key, count in (group_meta["label_counter"]).items():
            projected = float(stats["label_counter"][key]) + float(count)
            target = float(target_meta["label_counter"].get(key, 0.0))
            label_penalty += abs(projected - target) / max(1.0, target if target > 0 else 1.0)

        domain_penalty = 0.0
        for key, count in (group_meta["domain_counter"]).items():
            projected = float(stats["domain_counter"][key]) + float(count)
            target = float(target_meta["domain_counter"].get(key, 0.0))
            domain_penalty += abs(projected - target) / max(1.0, target if target > 0 else 1.0)

        hard_penalty = 0.0
        for key, count in (group_meta["hardness_counter"]).items():
            projected = float(stats["hardness_counter"][key]) + float(count)
            target = float(target_meta["hardness_counter"].get(key, 0.0))
            hard_penalty += abs(projected - target) / max(1.0, target if target > 0 else 1.0)

        proj_multi = float(stats["multi_label_rows"]) + float(group_meta["multi_label_rows"])
        proj_abstain = float(stats["abstain_rows"]) + float(group_meta["abstain_rows"])
        proj_novel = float(stats["novel_rows"]) + float(group_meta["novel_rows"])
        multi_penalty = abs(proj_multi - float(target_meta["multi_label_rows"])) / max(1.0, float(target_meta["multi_label_rows"]) if float(target_meta["multi_label_rows"]) > 0 else 1.0)
        abstain_penalty = abs(proj_abstain - float(target_meta["abstain_rows"])) / max(1.0, float(target_meta["abstain_rows"]) if float(target_meta["abstain_rows"]) > 0 else 1.0)
        novel_penalty = abs(proj_novel - float(target_meta["novel_rows"])) / max(1.0, float(target_meta["novel_rows"]) if float(target_meta["novel_rows"]) > 0 else 1.0)

        return (
            0.28 * row_penalty
            + 0.20 * label_penalty
            + 0.14 * domain_penalty
            + 0.10 * hard_penalty
            + 0.08 * multi_penalty
            + 0.08 * abstain_penalty
            + 0.12 * novel_penalty
        )

    def _apply(split: str, group_key_value: str) -> None:
        split_group_ids[split].append(group_key_value)
        stats = split_stats[split]
        group_meta = group_stats[group_key_value]
        stats["rows"] += int(group_meta["rows"])
        stats["label_counter"].update(group_meta["label_counter"])
        stats["domain_counter"].update(group_meta["domain_counter"])
        stats["hardness_counter"].update(group_meta["hardness_counter"])
        stats["multi_label_rows"] += int(group_meta["multi_label_rows"])
        stats["abstain_rows"] += int(group_meta["abstain_rows"])
        stats["novel_rows"] += int(group_meta["novel_rows"])

    def _remove(split: str, group_key_value: str) -> None:
        if group_key_value not in split_group_ids[split]:
            return
        split_group_ids[split].remove(group_key_value)
        stats = split_stats[split]
        group_meta = group_stats[group_key_value]
        stats["rows"] -= int(group_meta["rows"])
        stats["label_counter"].subtract(group_meta["label_counter"])
        stats["domain_counter"].subtract(group_meta["domain_counter"])
        stats["hardness_counter"].subtract(group_meta["hardness_counter"])
        stats["multi_label_rows"] -= int(group_meta["multi_label_rows"])
        stats["abstain_rows"] -= int(group_meta["abstain_rows"])
        stats["novel_rows"] -= int(group_meta["novel_rows"])

    rng = random.Random(random_seed)
    seed_tiebreakers = {group_name: rng.random() for group_name in groups}
    ordered_groups = sorted(
        groups,
        key=lambda group_name: (
            -int(group_stats[group_name]["rows"]),
            -sum(int(v) for v in group_stats[group_name]["label_counter"].values()),
            str(group_stats[group_name]["dominant_domain"]),
            seed_tiebreakers[group_name],
            group_name,
        ),
    )

    for group in ordered_groups:
        candidate_splits = ["train", "val", "test"]
        # Keep val/test alive when possible.
        if val_ratio > 0 and not split_group_ids["val"]:
            candidate_splits = ["val", "train", "test"]
        if test_ratio > 0 and not split_group_ids["test"]:
            candidate_splits = ["test", "train", "val"]
        split_scores = sorted(((_score_assignment(split, group), split) for split in candidate_splits), key=lambda item: (item[0], item[1]))
        _apply(split_scores[0][1], group)

    if len(groups) >= 3 and val_ratio > 0 and not split_group_ids["val"]:
        moved = split_group_ids["train"][-1] if len(split_group_ids["train"]) > 1 else (split_group_ids["test"][-1] if split_group_ids["test"] else None)
        if moved:
            if moved in split_group_ids["train"]:
                _remove("train", moved)
            else:
                _remove("test", moved)
            _apply("val", moved)
    if len(groups) >= 3 and test_ratio > 0 and not split_group_ids["test"]:
        moved = split_group_ids["train"][-1] if len(split_group_ids["train"]) > 1 else (split_group_ids["val"][-1] if split_group_ids["val"] else None)
        if moved:
            if moved in split_group_ids["train"]:
                _remove("train", moved)
            else:
                _remove("val", moved)
            _apply("test", moved)

    train = [row for group in split_group_ids["train"] for row in by_group[group]]
    val = [row for group in split_group_ids["val"] for row in by_group[group]]
    test = [row for group in split_group_ids["test"] for row in by_group[group]]
    return train, val, test


def v7_split_pipeline(
    rows: List[DedupChecked],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    group_key: str = "group_id",
) -> List[SplitAssigned]:
    """V7-specific split pipeline using grouped_split logic but with typed contracts."""
    train_rows, val_rows, test_rows = grouped_split(
        [r.model_dump() for r in rows], # Temp convert to dict for existing grouped_split compat
        group_key=group_key,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Assign splits back to typed rows
    id_to_split = {}
    for r in train_rows: id_to_split[r["row_id"]] = "train"
    for r in val_rows: id_to_split[r["row_id"]] = "val"
    for r in test_rows: id_to_split[r["row_id"]] = "test"
    
    results = []
    for r in rows:
        results.append(SplitAssigned(
            **r.model_dump(),
            split=id_to_split.get(r.row_id, "none")
        ))
    return results

def grouped_leakage_report(train_or_all: List[dict], val: List[dict] | None = None, test: List[dict] | None = None, *, group_key: str = "group_id") -> dict[str, Any]:
    """Audit for group_id leakage across splits. Supports either (train, val, test) or (all_rows)."""
    if val is None and test is None:
        # Unified list mode
        rows = train_or_all
        train_rows = [r for r in rows if r.get("split") == "train"]
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]
    else:
        # Three-list mode
        train_rows = train_or_all
        val_rows = val or []
        test_rows = test or []

    def _get_groups(rows: List[dict]) -> set[str]:
        return set(str(r.get(group_key) or "") for r in rows if (r.get(group_key) or ""))

    train_groups = _get_groups(train_rows)
    val_groups = _get_groups(val_rows)
    test_groups = _get_groups(test_rows)
    
    leakage_train_val = train_groups & val_groups
    leakage_train_test = train_groups & test_groups
    leakage_val_test = val_groups & test_groups
    
    total_leakage = len(leakage_train_val) + len(leakage_train_test) + len(leakage_val_test)
    
    return {
        "leakage_ok": total_leakage == 0,
        "total_leaked_groups": total_leakage,
        "overlap_counts": {
            "train_val": len(leakage_train_val),
            "train_test": len(leakage_train_test),
            "val_test": len(leakage_val_test)
        }
    }
