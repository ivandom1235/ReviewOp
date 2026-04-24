from __future__ import annotations

from typing import Any


def check_group_leakage(splits: dict[str, list[Any]]) -> dict[str, int | bool]:
    groups_by_split = {
        split: {str(getattr(row, "group_id", row.get("group_id") if isinstance(row, dict) else "")) for row in rows}
        for split, rows in splits.items()
    }
    leakage = 0
    names = list(groups_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            leakage += len(groups_by_split[left] & groups_by_split[right])
    return {"leakage_ok": leakage == 0, "grouped_leakage": leakage}


def check_text_duplication(splits: dict[str, list[Any]]) -> dict[str, int | bool]:
    seen: dict[str, str] = {}
    leakage = 0
    for split, rows in splits.items():
        for row in rows:
            text = str(getattr(row, "review_text", row.get("review_text") if isinstance(row, dict) else "")).strip().lower()
            if not text:
                continue
            prior = seen.get(text)
            if prior is not None and prior != split:
                leakage += 1
            seen[text] = split
    return {"leakage_ok": leakage == 0, "exact_text_leakage": leakage}


def check_cross_split_leakage(splits: dict[str, list[Any]]) -> dict[str, int]:
    """Check for both group and text leakage across splits."""
    g_results = check_group_leakage(splits)
    t_results = check_text_duplication(splits)
    return {
        "grouped_leakage": int(g_results.get("grouped_leakage", 0)),
        "exact_text_leakage": int(t_results.get("exact_text_leakage", 0))
    }
