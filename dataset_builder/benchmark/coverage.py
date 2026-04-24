from __future__ import annotations


def check_anchor_coverage(rows: list[object], minimum: int = 1) -> bool:
    return len(rows) >= minimum


def repair_anchor_coverage(rows: list[object]) -> list[object]:
    return rows
