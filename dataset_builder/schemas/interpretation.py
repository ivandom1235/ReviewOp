from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


VALID_LABEL_TYPES = {"explicit", "implicit", "verified"}


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

    def __post_init__(self) -> None:
        if self.label_type not in VALID_LABEL_TYPES:
            raise ValueError(f"invalid label_type: {self.label_type}")
        if not str(self.aspect_raw).strip():
            raise ValueError("aspect_raw is required")
        if not str(self.aspect_canonical).strip():
            raise ValueError("aspect_canonical is required")
        if len(self.evidence_span) != 2:
            raise ValueError("evidence_span must be [start, end]")
        if self.canonical_confidence < 0 or self.canonical_confidence > 1:
            raise ValueError("canonical_confidence must be in [0, 1]")
