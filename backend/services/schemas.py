# proto/backend/services/implicit/schemas.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ImplicitPredictRequest(BaseModel):
    review_text: str = Field(..., min_length=1)
    allowed_aspects: Optional[List[str]] = None
    max_predictions_per_sentence: int = Field(default=2, ge=1, le=10)
    max_predictions_per_review: int = Field(default=5, ge=1, le=20)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ImplicitCandidateSchema(BaseModel):
    aspect: str
    confidence: float
    sentiment_hint: str
    evidence_sentence: str
    sentence_index: int
    domain_hint: Optional[str] = None
    support_count: Optional[int] = 1
    decision: Optional[str] = None
    routing: Optional[str] = None
    ambiguity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    decision_band: Optional[Literal["known", "boundary", "novel"]] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None


class SelectivePredictionSchema(BaseModel):
    aspect: str
    sentiment: str = "neutral"
    confidence: float
    routing: Literal["known", "novel"] = "known"
    evidence_text: Optional[str] = None
    evidence_span: Optional[List[int]] = None
    decision_band: Optional[Literal["known", "boundary", "novel"]] = None
    novelty_score: Optional[float] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None


class AbstainedPredictionSchema(BaseModel):
    reason: str
    confidence: float
    ambiguity_score: float


class NovelCandidateSchema(BaseModel):
    aspect: str
    novelty_score: float
    confidence: Optional[float] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None
    evidence_text: Optional[str] = None


class ImplicitPredictResponse(BaseModel):
    review_text: str
    sentences: List[str]
    implicit_predictions: List[ImplicitCandidateSchema]
    accepted_predictions: List[SelectivePredictionSchema] = Field(default_factory=list)
    abstained_predictions: List[AbstainedPredictionSchema] = Field(default_factory=list)
    novel_candidates: List[NovelCandidateSchema] = Field(default_factory=list)


class ExplicitEvidenceSpanSchema(BaseModel):
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    snippet: str = ""
    source: Optional[str] = None


class UnifiedCandidateSchema(BaseModel):
    aspect: str
    source: Literal["explicit", "implicit", "merged"]
    confidence: float
    sentiment_hint: Optional[str] = None
    evidence_sentences: List[str] = []
    evidence_spans: List[Dict[str, Any]] = []
    domains: List[str] = []
    support_count: int = 1


class VerifierPayloadSchema(BaseModel):
    review_text: str
    num_candidates: int
    candidates: List[UnifiedCandidateSchema]


class DebugReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=1)
    explicit_candidates: Optional[List[Dict[str, Any]]] = None
    allowed_aspects: Optional[List[str]] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_predictions_per_sentence: int = Field(default=2, ge=1, le=10)
    max_predictions_per_review: int = Field(default=5, ge=1, le=20)
    max_merged_candidates: int = Field(default=10, ge=1, le=50)


class DebugReviewResponse(BaseModel):
    review_text: str
    sentences: List[str]
    explicit_candidates: List[Dict[str, Any]]
    implicit_predictions: List[ImplicitCandidateSchema]
    verifier_payload: VerifierPayloadSchema


class HealthResponse(BaseModel):
    status: str = "ok"
    module: str = "implicit"
    checkpoint_loaded: bool
    num_labels: int
