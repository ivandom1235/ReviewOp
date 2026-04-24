from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


VALID_LABEL_TYPES = {"explicit", "implicit"}
VALID_SOURCE_TYPES = {"explicit", "implicit_learned", "implicit_json", "implicit_llm", "merged"}


@dataclass(frozen=True)
class Interpretation:
    aspect_raw: str
    latent_family: str
    aspect_canonical: str
    label_type: str
    sentiment: str
    evidence_text: str
    evidence_span: list[int]
    source: str
    support_type: str
    canonical_confidence: float = 1.0
    mapping_source: str = "none"
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
    repair_severity: int = 0
    matched_pattern: Optional[str] = None
    pattern_id: Optional[str] = None
    source_type: str = "unknown"
    pattern_confidence: Optional[float] = None
    evidence_scope: str = "unknown"
    novelty_status: str = "unknown"
    novelty_score: Optional[float] = None
    novelty_reason: Optional[str] = None
    abstain_acceptable: Optional[bool] = None
    aspect_anchor: Optional[str] = None
    modifier_terms: tuple[str, ...] = field(default_factory=tuple)
    anchor_source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.aspect_anchor and not self.anchor_source:
            raise ValueError("anchor_source is required when aspect_anchor is present")
        if self.label_type not in VALID_LABEL_TYPES:
            raise ValueError(f"invalid label_type: {self.label_type}")
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"invalid source_type: {self.source_type}")
        if self.label_type == "explicit" and self.source_type != "explicit":
            raise ValueError("explicit label_type requires source_type='explicit'")
        if self.label_type == "implicit" and self.source_type == "explicit":
            raise ValueError("implicit label_type cannot use source_type='explicit'")
        if self.source_type == "implicit_learned":
            if not str(self.matched_pattern or "").strip():
                raise ValueError("implicit_learned requires matched_pattern")
            if not str(self.pattern_id or "").strip():
                raise ValueError("implicit_learned requires pattern_id")
        if self.source_type == "explicit" and (self.matched_pattern or self.pattern_id):
            raise ValueError("explicit interpretations cannot include pattern metadata")
        if not str(self.aspect_raw).strip():
            raise ValueError("aspect_raw is required")
        if not str(self.aspect_canonical).strip():
            raise ValueError("aspect_canonical is required")
        if len(self.evidence_span) != 2:
            raise ValueError("evidence_span must be [start, end]")
        if self.canonical_confidence < 0 or self.canonical_confidence > 1:
            raise ValueError("canonical_confidence must be in [0, 1]")
