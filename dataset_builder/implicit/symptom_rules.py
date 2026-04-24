from __future__ import annotations

from .symptom_store import SymptomPatternStore


def extract_symptom_candidates(text: str, pattern_store: SymptomPatternStore | None = None, domain: str | None = None) -> list[str]:
    if pattern_store is None:
        return []
    return pattern_store.matching_canonicals(text, domain=domain)
