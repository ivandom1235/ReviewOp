from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, List, Optional
from pydantic import BaseModel, ConfigDict, Field


RowLifecycleState = Literal[
    "raw_loaded",
    "prepared",
    "implicit_scored",
    "grounded",
    "quality_scored",
    "bucketed",
    "dedup_checked",
    "split_assigned",
    "benchmark_gold",
    "benchmark_silver",
    "train_keep",
    "review_queue",
    "hard_reject",
    "promoted_to_train",
    "promoted_to_benchmark",
]


@dataclass(slots=True)
class RowLifecycleRecord:
    row_id: str
    state: RowLifecycleState
    reason_codes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QualityDecision:
    decision: Literal["benchmark_gold", "benchmark_silver", "train_keep", "review_queue", "hard_reject"]
    quality_score: float
    usefulness_score: float
    redundancy_score: float
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecoveryDecision:
    eligible: bool
    selected: bool
    reason_codes: list[str] = field(default_factory=list)
    source_state: RowLifecycleState = "review_queue"
    target_state: RowLifecycleState | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StageProfile:
    stage_name: str
    rows_in: int = 0
    rows_out: int = 0
    elapsed_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineProfile:
    stages: list[StageProfile] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- V7 Pydantic State Machine Models ---

class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect: str
    sentiment: str
    evidence_text: Optional[str] = None
    evidence_span: Optional[List[int]] = None
    interpretation_type: Literal["explicit", "implicit"] = "explicit"
    confidence: float = 1.0

class GroundedInterpretation(Interpretation):
    evidence_text: str # Mandatory for grounded state
    evidence_span: List[int] # Mandatory offsets [start, end]

class RowBase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    row_id: str
    review_text: str
    domain: Optional[str] = None
    group_id: Optional[str] = None

class RawLoaded(BaseModel):
    source_id: str
    review_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class Prepared(RowBase):
    domain: str
    group_id: str

class ImplicitScored(Prepared):
    interpretations: List[Interpretation] = Field(default_factory=list)

class Grounded(Prepared):
    interpretations: List[GroundedInterpretation]

class QualityScored(RowBase):
    interpretations: List[Interpretation] # Can be grounded or implicit
    quality_score: float
    is_v7_gold: bool = False
    reason: str = "unknown"
    usefulness_score: float = 0.0
    redundancy_score: float = 0.0
    abstain_acceptable: bool = False
    abstain_reason: Optional[Literal["insufficient_evidence", "competing_interpretations", "weak_domain_signal", "novel_but_unstable"]] = None

class BucketAssigned(QualityScored):
    bucket: Literal["benchmark_gold", "benchmark_silver", "train_keep", "review_queue", "hard_reject"]
    promotion_reason: Optional[str] = None

class DedupChecked(BucketAssigned):
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    cluster_id: Optional[str] = None
    dedup_strategy: Optional[Literal["exact", "fuzzy", "semantic"]] = None

class SplitAssigned(DedupChecked):
    split: Literal["train", "val", "test", "none"]

class BenchmarkInstance(BaseModel):
    instance_id: str
    review_text: str
    domain: str
    group_id: str
    domain_family: str = "unknown"
    annotation_source: str = "extraction"
    gold_interpretations: List[GroundedInterpretation]
    hardness_tier: Literal["H0", "H1", "H2", "H3"] = "H0"
    is_ambiguous: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

class TrainExample(BaseModel):
    example_id: str
    review_text: str
    domain: str
    domain_family: str = "unknown"
    interpretations: List[Interpretation]
    support_type: str = "grounded"
    metadata: dict[str, Any] = Field(default_factory=dict)

class NoveltyRecord(BaseModel):
    row_id: str
    novel_cluster_id: str
    alias: Optional[str] = None
    evidence_summary: str
