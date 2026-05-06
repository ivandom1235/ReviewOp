from __future__ import annotations

from typing import Any


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def check_group_leakage(splits: dict[str, list[Any]]) -> dict[str, int | bool]:
    groups_by_split = {}
    for split, rows in splits.items():
        s_groups = set()
        for row in rows:
            gid = str(_get_val(row, "group_id", "")).strip()
            if gid:
                s_groups.add(gid)
        groups_by_split[split] = s_groups
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


def check_near_duplicates(splits: dict[str, list[Any]], threshold: float = 0.9) -> dict[str, int]:
    from rapidfuzz import fuzz
    texts_by_split = {}
    for split, rows in splits.items():
        texts_by_split[split] = [
            str(getattr(row, "review_text", row.get("review_text") if isinstance(row, dict) else "")).strip().lower()
            for row in rows
        ]
    
    leakage = 0
    names = list(texts_by_split)
    for i, left_name in enumerate(names):
        left_texts = texts_by_split[left_name]
        for right_name in names[i+1:]:
            right_texts = texts_by_split[right_name]
            for lt in left_texts:
                if not lt: continue
                for rt in right_texts:
                    if not rt: continue
                    # Avoid exact match which is handled by check_text_duplication
                    if lt == rt: continue
                    score = fuzz.ratio(lt, rt) / 100.0
                    if score >= threshold:
                        leakage += 1
                        break # Only count once per left-text to avoid explosion
    return {"near_duplicate_leakage": leakage}


def check_cross_split_leakage(splits: dict[str, list[Any]]) -> dict[str, int]:
    """Check for both group and text leakage across splits."""
    g_results = check_group_leakage(splits)
    t_results = check_text_duplication(splits)
    n_results = check_near_duplicates(splits)
    return {
        "grouped_leakage": int(g_results.get("grouped_leakage", 0)),
        "exact_text_leakage": int(t_results.get("exact_text_leakage", 0)),
        "near_duplicate_leakage": int(n_results.get("near_duplicate_leakage", 0))
    }
