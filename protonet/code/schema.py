from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoldInterpretation:
    aspect: str
    label_type: str
    sentiment: str | None
    evidence_text: str
    evidence_span: list[int] | None = None
    evidence_scope: str = "unknown"
    novelty_status: str = "known"
    mapping_scope: str | None = None
    confidence: float | None = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewExample:
    row_id: str
    review_id: str
    text: str
    domain: str
    split: str
    source_type: str
    novelty_status: str
    abstain_acceptable: bool
    abstain_reason_gold: list[str]
    gold_interpretations: list[GoldInterpretation]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ECScore:
    aspect: str
    proto_similarity: float
    evidence_support: float = 0.0
    memory_support: float = 0.0
    verifier_support: float = 0.0
    ambiguity_penalty: float = 0.0
    novelty_risk: float = 0.0
    contradiction_score: float = 0.0
    final_score: float = 0.0
    decision: str = "unknown"
    decision_reason: str | None = None
    prototype_source: str | None = None
