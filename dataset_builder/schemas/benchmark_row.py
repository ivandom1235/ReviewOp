from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .interpretation import Interpretation


@dataclass(frozen=True)
class BenchmarkRow:
    review_id: str
    group_id: str
    domain: str
    domain_family: str
    review_text: str
    explicit_interpretations: list[Interpretation] = field(default_factory=list)
    implicit_interpretations: list[Interpretation] = field(default_factory=list)
    gold_interpretations: list[Interpretation] = field(default_factory=list)
    ambiguity_score: float = 0.0
    novelty_status: str = "known"
    abstain_acceptable: bool = False
    hardness_tier: str = "H0"
    provenance: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.group_id).strip():
            raise ValueError("group_id is required")
