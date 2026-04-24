from __future__ import annotations

from .symptom_rules import extract_symptom_candidates
from .symptom_store import SymptomPatternStore


def weak_label_reviews(
    texts: list[str],
    pattern_store: SymptomPatternStore | None = None,
    domains: list[str] | None = None,
) -> list[dict[str, object]]:
    domains = domains or ["unknown"] * len(texts)
    return [
        {
            "text": text,
            "domain": domain,
            "candidates": extract_symptom_candidates(text, pattern_store, domain=domain),
            "provenance": "learned_symptom_patterns",
        }
        for text, domain in zip(texts, domains)
    ]
