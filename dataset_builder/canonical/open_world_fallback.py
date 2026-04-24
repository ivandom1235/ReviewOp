from __future__ import annotations


def keep_open_world_candidate(candidate: str, confidence: float) -> bool:
    return bool(str(candidate or "").strip()) and float(confidence) >= 0.0


def mark_provisional_canonical(candidate: str) -> str:
    return str(candidate or "").strip().lower().replace(" ", "_")
