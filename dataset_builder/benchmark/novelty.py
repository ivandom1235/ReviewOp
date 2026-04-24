from __future__ import annotations


def detect_novelty(aspect_canonical: str, known_aspects: set[str]) -> str:
    return "known" if aspect_canonical in known_aspects else "novel"


def balance_novelty_across_splits(splits: dict[str, list[object]]) -> dict[str, int]:
    return {split: len(rows) for split, rows in splits.items()}
