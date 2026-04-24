from __future__ import annotations

from dataclasses import replace

from ..canonical.domain_registry import DomainRegistry
from .symptom_store import SymptomPatternCandidate


BROAD_SYMPTOM_PHRASES = {"good", "bad", "quality", "works", "service"}


def validate_symptom_patterns(
    candidates: list[SymptomPatternCandidate],
    *,
    min_support: int = 2,
    min_evidence_valid_rate: float = 0.8,
    min_precision_estimate: float = 0.75,
    domain: str | None = None
) -> list[SymptomPatternCandidate]:
    broad_phrases = set(DomainRegistry.get_broad_labels(domain))
    validated: list[SymptomPatternCandidate] = []
    for candidate in candidates:
        reasons: list[str] = []
        if candidate.support_count < min_support:
            reasons.append("insufficient_support")
        if candidate.evidence_valid_rate < min_evidence_valid_rate:
            reasons.append("invalid_evidence")
        if candidate.precision_estimate < min_precision_estimate:
            reasons.append("low_precision")
        if candidate.phrase in broad_phrases:
            reasons.append("broad_phrase")

        status = "review_queue" if reasons else "promoted"
        domain_scope = "global" if len(candidate.domains) > 1 else "domain_scoped"
        validated.append(
            replace(
                candidate,
                status=status,
                domain_scope=domain_scope,
                reason_codes=tuple(reasons),
            )
        )
    return validated
