from __future__ import annotations


def build_novelty_report(splits: dict[str, list[object]]) -> dict[str, int]:
    return {
        split: sum(1 for row in rows if getattr(row, "novelty_status", "known") != "known")
        for split, rows in splits.items()
    }
