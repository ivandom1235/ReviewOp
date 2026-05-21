from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


@dataclass(frozen=True)
class GoldAspect:
    aspect: str
    label_type: str = "unknown"
    sentiment: str | None = None
    evidence_text: str = ""
    evidence_span: list[int] | None = None
    evidence_scope: str = "unknown"
    novelty_status: str = "known"
    mapping_scope: str | None = None
    mapping_layers: list[str] = field(default_factory=list)
    confidence: float | None = None
    matched_terms: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewExample:
    row_id: str
    review_id: str
    text: str
    domain: str
    split: str
    source_type: str
    query_text: str = ""
    novelty_status: str = "known"
    abstain_acceptable: bool = False
    abstain_reason_gold: list[str] = field(default_factory=list)
    gold_aspects: list[GoldAspect] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    evidence_text: str = ""
    group_id: str | None = None
    parent_review_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    gold_unseen_labels: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_labels(self) -> set[str]:
        return {g.aspect for g in self.gold_aspects if g.aspect}

    @property
    def gold_emerging_labels(self) -> set[str]:
        return {g.aspect for g in self.gold_aspects if g.novelty_status == "novel"}

    @property
    def gold_novel_labels(self) -> set[str]:
        # Backward-compatible: dataset novelty status, not true unseen class.
        return self.gold_emerging_labels

    @property
    def gold_boundary_labels(self) -> set[str]:
        return {g.aspect for g in self.gold_aspects if g.novelty_status == "boundary"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeExample:
    row_id: str
    review_id: str
    text: str
    domain: str
    split: str
    source_type: str
    group_id: str | None = None
    parent_review_id: str | None = None


def to_runtime_example(ex: ReviewExample) -> RuntimeExample:
    return RuntimeExample(
        row_id=ex.row_id,
        review_id=ex.review_id,
        text=ex.text,
        domain=ex.domain,
        split=ex.split,
        source_type=ex.source_type,
        group_id=ex.group_id,
        parent_review_id=ex.parent_review_id,
    )


@dataclass(frozen=True)
class CandidateScore:
    aspect: str
    proto_score: float
    final_score: float
    rank: int
    evidence_support: float = 0.0
    memory_support: float = 0.0
    description_support: float = 0.0
    margin_to_next: float = 0.0
    novelty_risk: float = 0.0
    ambiguity_risk: float = 0.0
    evidence_scope_score: float = 0.0
    support_count: int = 0
    prototype_source: str = "train_evidence"
    candidate_type: str = "known"  # known | open_world
    candidate_source: str = "prototype"  # prototype | classifier | memory | hybrid
    source_aspect: str = ""
    known_confidence: float = 0.0
    lexical_evidence_support: float = 0.0
    semantic_evidence_support: float = 0.0
    open_world_evidence_quality: float = 0.0
    residual_score: float = 0.0
    energy_score: float = 0.0
    unknown_score: float = 0.0
    decision: str = "unrouted"
    decision_reason: str = "not_routed"
    reranker_score: float = 0.0

    def with_decision(self, decision: str, reason: str) -> "CandidateScore":
        return CandidateScore(
            aspect=self.aspect,
            proto_score=self.proto_score,
            final_score=self.final_score,
            rank=self.rank,
            evidence_support=self.evidence_support,
            description_support=self.description_support,
            memory_support=self.memory_support,
            margin_to_next=self.margin_to_next,
            novelty_risk=self.novelty_risk,
            ambiguity_risk=self.ambiguity_risk,
            evidence_scope_score=self.evidence_scope_score,
            support_count=self.support_count,
            prototype_source=self.prototype_source,
            candidate_type=self.candidate_type,
            candidate_source=self.candidate_source,
            source_aspect=self.source_aspect,
            known_confidence=self.known_confidence,
            lexical_evidence_support=self.lexical_evidence_support,
            semantic_evidence_support=self.semantic_evidence_support,
            open_world_evidence_quality=self.open_world_evidence_quality,
            residual_score=self.residual_score,
            energy_score=self.energy_score,
            unknown_score=self.unknown_score,
            decision=decision,
            decision_reason=reason,
            reranker_score=self.reranker_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    row_id: str
    review_id: str
    split: str
    domain: str
    text: str
    gold_labels: list[str]
    gold_novel_labels: list[str]
    gold_boundary_labels: list[str]
    abstain_acceptable: bool
    candidates: list[CandidateScore]
    gold_emerging_labels: list[str] = field(default_factory=list)
    gold_unseen_labels: list[str] = field(default_factory=list)

    def accepted_candidates(self, accepted_decisions: tuple[str, ...] = ("accept_known",)) -> list[CandidateScore]:
        return [c for c in self.candidates if c.decision in set(accepted_decisions)]

    def predicted_labels(self, accepted_decisions: tuple[str, ...] = ("accept_known",)) -> list[str]:
        return [c.aspect for c in self.accepted_candidates(accepted_decisions)]

    @property
    def review_candidates(self) -> list[CandidateScore]:
        return [c for c in self.candidates if c.decision == "needs_review"]

    @property
    def has_abstain(self) -> bool:
        return any(c.decision == "abstain" for c in self.candidates[:1])

    @property
    def has_open_world(self) -> bool:
        return bool(
            self.candidates
            and self.candidates[0].decision in {"open_world_candidate", "named_open_world_candidate"}
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predicted_labels"] = self.predicted_labels()
        d["has_abstain"] = self.has_abstain
        d["has_open_world"] = self.has_open_world
        d["gold_emerging_labels"] = self.gold_emerging_labels
        d["gold_unseen_labels"] = self.gold_unseen_labels
        return d


def gold_from_raw(g: dict[str, Any], fallback_text: str = "", default_novelty_status: str = "known") -> GoldAspect:
    aspect = (
        g.get("aspect_canonical")
        or g.get("canonical_aspect")
        or g.get("aspect")
        or g.get("label")
        or g.get("aspect_raw")
        or "unknown"
    )
    evidence_text = g.get("evidence_text") or g.get("evidence") or g.get("snippet") or fallback_text
    mapping_layers = g.get("mapping_layers") or []
    if isinstance(mapping_layers, str):
        mapping_layers = [mapping_layers]
    matched_terms = g.get("matched_terms") or []
    if isinstance(matched_terms, str):
        matched_terms = [matched_terms]
    conf = g.get("canonical_confidence", g.get("confidence"))
    try:
        conf = None if conf is None else float(conf)
    except Exception:
        conf = None
    return GoldAspect(
        aspect=_str(aspect, "unknown"),
        label_type=_str(g.get("label_type", g.get("source_type", "unknown")), "unknown"),
        sentiment=g.get("sentiment"),
        evidence_text=_str(evidence_text, fallback_text),
        evidence_span=g.get("evidence_span"),
        evidence_scope=_str(g.get("evidence_scope", "unknown"), "unknown"),
        novelty_status=_str(g.get("novelty_status") or default_novelty_status or "known", "known"),
        mapping_scope=g.get("mapping_scope"),
        mapping_layers=list(mapping_layers),
        confidence=conf,
        matched_terms=list(matched_terms),
        raw=g,
    )


def example_from_raw(row: dict[str, Any], split: str | None = None) -> ReviewExample:
    text = _str(row.get("review_text") or row.get("text") or row.get("sentence") or "")
    query_text = _str(row.get("query_text") or row.get("evidence_text") or row.get("sentence") or text)
    row_novelty = _str(row.get("novelty_status", "known"), "known")
    raw_gold = row.get("gold_interpretations") or row.get("gold_aspects") or row.get("labels") or []
    if isinstance(raw_gold, dict):
        raw_gold = [raw_gold]
    gold = [gold_from_raw(g, fallback_text=text, default_novelty_status=row_novelty) for g in raw_gold]
    matched_terms = sorted({str(term).strip() for g in gold for term in g.matched_terms if str(term).strip()})
    evidence_text = next((g.evidence_text for g in gold if str(g.evidence_text or "").strip()), text)
    abstain_acceptable = bool(row.get("abstain_acceptable", False))
    abstain_reason_gold = row.get("abstain_reason_gold") or row.get("reason_gold") or []
    if isinstance(abstain_reason_gold, str):
        abstain_reason_gold = [abstain_reason_gold]
    if abstain_acceptable and not abstain_reason_gold:
        abstain_reason_gold = ["unspecified_abstain_reason"]
    return ReviewExample(
        row_id=_str(row.get("row_id") or row.get("id") or row.get("example_id") or "unknown"),
        review_id=_str(row.get("review_id") or row.get("parent_review_id") or row.get("row_id") or "unknown"),
        text=text,
        query_text=query_text,
        domain=_str(row.get("domain", "unknown"), "unknown"),
        split=_str(split or row.get("split", "unknown"), "unknown"),
        source_type=_str(row.get("source_type") or row.get("label_source") or "unknown", "unknown"),
        novelty_status=row_novelty,
        abstain_acceptable=abstain_acceptable,
        abstain_reason_gold=list(abstain_reason_gold),
        gold_aspects=gold,
        matched_terms=matched_terms,
        evidence_text=evidence_text,
        group_id=row.get("group_id"),
        parent_review_id=row.get("parent_review_id"),
        metadata=row.get("metadata", {}),
        raw=row,
    )


ASPECT_PARENT = {
    "food_quality": "quality",
    "service_quality": "service",
    "service_speed": "service",
    "customer_support": "service",
    "delivery": "service",
    "storage": "hardware",
    "keyboard": "hardware",
    "display": "hardware",
    "battery_life": "hardware",
    "power": "hardware",
    "portability": "hardware",
    "trackpad": "hardware",
    "performance": "system_experience",
    "software": "system_experience",
    "usability": "system_experience",
    "connectivity": "system_experience",
    "call_reliability": "system_experience",
    "price": "value",
    "value": "value",
    "ambience": "experience",
    "aesthetics": "experience",
}


def parent_of(aspect: str) -> str:
    if not aspect:
        return "unknown"
    return ASPECT_PARENT.get(aspect, aspect)

