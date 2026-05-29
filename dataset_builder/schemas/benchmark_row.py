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
    row_id: str | None = None
    query_text: str = ""
    explicit_interpretations: list[Interpretation] = field(default_factory=list)
    implicit_interpretations: list[Interpretation] = field(default_factory=list)
    gold_interpretations: list[Interpretation] = field(default_factory=list)
    ambiguity_score: float = 0.0
    novelty_status: str = "known"
    novelty_score: float = 0.0
    abstain_acceptable: bool = False
    abstain_reason_gold: tuple[str, ...] = field(default_factory=tuple)
    ambiguity_level: str = "low"
    hardness_tier: str = "H0"
    source_type: str = "unknown"
    pattern_id: str | None = None
    mapping_source: str = "unknown"
    mapping_scope: str = "unknown"
    row_source_type: str = "unknown"
    row_mapping_scope: str = "unknown"
    row_mapping_sources: tuple[str, ...] = field(default_factory=tuple)
    counterfactual_group_id: str | None = None
    counterfactual_role: str | None = None
    counterfactual_source_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    score_components: dict[str, float] = field(default_factory=dict)
    candidate_trace: dict[str, list[str]] = field(default_factory=lambda: {
        "after_extraction": [],
        "after_fusion": [],
        "after_canonicalization": [],
        "after_pruning": []
    })
    split_protocol: dict[str, str] = field(default_factory=lambda: {"random": "unused", "grouped": "unused", "domain_holdout": "unused"})

    def __post_init__(self) -> None:
        if not str(self.group_id).strip():
            raise ValueError("group_id is required")
