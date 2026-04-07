from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class EvidenceSpanOut(BaseModel):
    start_char: int
    end_char: int
    snippet: str


class PredictionOut(BaseModel):
    aspect_raw: str
    aspect_cluster: str
    sentiment: str  # positive|neutral|negative
    confidence: float

    aspect_weight: Optional[float] = None
    aspect_score: Optional[float] = None

    evidence_spans: List[EvidenceSpanOut] = Field(default_factory=list)
    rationale: Optional[str] = None
    source: Optional[str] = None
    is_implicit: Optional[bool] = None
    verification_status: Optional[str] = None
    decision: Optional[str] = None
    routing: Optional[str] = None
    ambiguity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    decision_band: Optional[Literal["known", "boundary", "novel"]] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None


class SelectivePredictionOut(BaseModel):
    aspect: str
    sentiment: str = "neutral"
    confidence: float
    routing: str = "known"
    evidence: Optional[str] = None
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    decision_band: Optional[Literal["known", "boundary", "novel"]] = None
    novelty_score: Optional[float] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None


class AbstainedPredictionOut(BaseModel):
    reason: str
    confidence: float
    ambiguity_score: float


class NovelCandidateOut(BaseModel):
    aspect: str
    novelty_score: float
    confidence: Optional[float] = None
    novel_cluster_id: Optional[str] = None
    novel_alias: Optional[str] = None
    evidence_text: Optional[str] = None


class InferReviewIn(BaseModel):
    text: str
    domain: Optional[str] = None
    product_id: Optional[str] = None


class InferReviewOut(BaseModel):
    review_id: int
    domain: Optional[str] = None
    product_id: Optional[str] = None
    predictions: List[PredictionOut]

    # NEW:
    overall_sentiment: Optional[str] = None
    overall_score: Optional[float] = None
    overall_confidence: Optional[float] = None
    accepted_predictions: List[SelectivePredictionOut] = Field(default_factory=list)
    abstained_predictions: List[AbstainedPredictionOut] = Field(default_factory=list)
    novel_candidates: List[NovelCandidateOut] = Field(default_factory=list)


class JobCreateOut(BaseModel):
    job_id: str
    status: str
    total: int


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    total: int
    processed: int
    failed: int
    error: Optional[str] = None


class OverviewOut(BaseModel):
    total_reviews: int
    total_aspect_mentions: int
    unique_aspects_raw: int
    avg_confidence: float
    sentiment_counts: dict


class TopAspectOut(BaseModel):
    aspect: str
    count: int


class AspectSentimentDistOut(BaseModel):
    aspect: str
    positive: int
    neutral: int
    negative: int


class TrendPointOut(BaseModel):
    bucket: str
    mentions: int
    negative_pct: float
    sentiment_score: float


class DashboardKpiOut(BaseModel):
    total_reviews: int
    total_aspects: int
    most_negative_aspect: Optional[str] = None
    negative_sentiment_pct: float = 0.0
    emerging_issues_count: int = 0


class AspectLeaderboardRowOut(BaseModel):
    aspect: str
    frequency: int
    sample_size: int = 0
    mentions_per_100_reviews: float = 0.0
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    net_sentiment: float
    change_vs_previous_period: float
    change_7d_vs_prev_7d: float = 0.0
    negative_ci_low: float = 0.0
    negative_ci_high: float = 0.0
    implicit_pct: float


class AspectTrendPointOut(BaseModel):
    bucket: str
    aspect: str
    mentions: int
    negative_pct: float


class EvidenceRowOut(BaseModel):
    review_id: int
    review_text: str
    aspect: str
    sentiment: str
    origin: str
    evidence: Optional[str] = None
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    created_at: Optional[str] = None


class AspectDetailOut(BaseModel):
    aspect: str
    frequency: int
    positive: int
    neutral: int
    negative: int
    explicit_count: int
    implicit_count: int
    connected_aspects: List[dict] = Field(default_factory=list)
    trend: List[AspectTrendPointOut] = Field(default_factory=list)
    examples: List[EvidenceRowOut] = Field(default_factory=list)


class AlertOut(BaseModel):
    id: int
    type: str
    aspect: str
    severity: str
    message: str
    value: float
    threshold: float


class UserReviewSummaryOut(BaseModel):
    total_user_reviews: int
    unique_reviewers: int
    average_rating: float
    recommendation_rate: float
    reviews_last_7_days: int
    top_products: List[dict] = Field(default_factory=list)


class AdminUserReviewRowOut(BaseModel):
    review_id: int
    product_id: str
    product_name: Optional[str] = None
    username: str
    rating: int
    recommendation: Optional[bool] = None
    helpful_count: int = 0
    review_title: Optional[str] = None
    review_text: str
    created_at: str


class UserReviewListOut(BaseModel):
    total: int
    limit: int
    offset: int
    rows: List[AdminUserReviewRowOut] = Field(default_factory=list)


class AdminExportOut(BaseModel):
    generated_at: str
    dashboard_kpis: DashboardKpiOut
    aspect_leaderboard: List[AspectLeaderboardRowOut] = Field(default_factory=list)
    aspect_trends: List[AspectTrendPointOut] = Field(default_factory=list)
    emerging_aspects: List[dict] = Field(default_factory=list)
    evidence: List[EvidenceRowOut] = Field(default_factory=list)
    alerts: List[AlertOut] = Field(default_factory=list)
    impact_matrix: List["ImpactMatrixRowOut"] = Field(default_factory=list)
    segments: List["SegmentDrilldownOut"] = Field(default_factory=list)
    weekly_summary: "WeeklySummaryOut"
    user_reviews_summary: UserReviewSummaryOut
    user_reviews: UserReviewListOut


class ImpactMatrixRowOut(BaseModel):
    aspect: str
    volume: int
    negative_rate: float
    growth_pct: float
    priority_score: float
    action_tier: str


class SegmentDrilldownOut(BaseModel):
    segment_type: str
    segment_value: str
    review_count: int
    mention_count: int
    negative_pct: float
    top_negative_aspect: Optional[str] = None


class WeeklySummaryOut(BaseModel):
    period_label: str
    top_drivers: List[str] = Field(default_factory=list)
    biggest_increase_aspect: Optional[str] = None
    biggest_increase_pct: float = 0.0
    emerging_count: int = 0
    action_recommendations: List[str] = Field(default_factory=list)


# Optional: KG payloads (for next step endpoints)
class AspectNodeOut(BaseModel):
    aspect_cluster: str
    domain: Optional[str] = None
    df: Optional[int] = None
    idf: Optional[float] = None
    centrality: Optional[float] = None


class AspectEdgeOut(BaseModel):
    src_aspect: str
    dst_aspect: str
    edge_type: str
    weight: float
    domain: Optional[str] = None

class CentralityOut(BaseModel):
    aspect: str
    centrality: float
    df: int = 0
    idf: float = 0.0


class CommunityOut(BaseModel):
    community_id: int
    aspects: List[str]


class EdgeOut(BaseModel):
    src: str
    dst: str
    edge_type: str
    weight: float


class GraphNodeOut(BaseModel):
    id: str
    label: str
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    frequency: Optional[int] = None
    avg_sentiment: Optional[float] = None
    dominant_sentiment: Optional[str] = None
    negative_ratio: Optional[float] = None
    explicit_count: int = 0
    implicit_count: int = 0
    evidence: Optional[str] = None
    evidence_start: Optional[int] = None
    evidence_end: Optional[int] = None
    origin: Optional[str] = None


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    directional: bool = False
    pair_count: Optional[int] = None
    polarity_hint: Optional[str] = None
    example_reviews: List[str] = Field(default_factory=list)


class GraphResponseOut(BaseModel):
    scope: str
    review_id: Optional[int] = None
    generated_at: Optional[str] = None
    filters: dict = Field(default_factory=dict)
    nodes: List[GraphNodeOut] = Field(default_factory=list)
    edges: List[GraphEdgeOut] = Field(default_factory=list)


class AuthRegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=5, max_length=120)


class AuthLoginIn(BaseModel):
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=5, max_length=120)


class AuthUserOut(BaseModel):
    id: int
    username: str
    role: Literal["admin", "user"]


class AuthLoginOut(BaseModel):
    token: str
    user: AuthUserOut


class ProductCardOut(BaseModel):
    product_id: str
    name: str
    category: Optional[str] = None
    summary: Optional[str] = None
    average_rating: float = 0.0
    review_count: int = 0
    latest_review_at: Optional[str] = None


class StarDistributionOut(BaseModel):
    stars: int
    count: int


class AspectSummaryOut(BaseModel):
    aspect: str
    sentiment: str


class ProductReviewOut(BaseModel):
    review_id: int
    product_id: str
    reviewer_name: str
    rating: int
    review_title: Optional[str] = None
    review_text: str
    review_date: str
    helpful_count: int = 0
    aspects: List[AspectSummaryOut] = Field(default_factory=list)


class ProductDetailOut(BaseModel):
    product_id: str
    name: str
    category: Optional[str] = None
    summary: Optional[str] = None
    average_rating: float = 0.0
    review_count: int = 0
    star_distribution: List[StarDistributionOut] = Field(default_factory=list)


class ProductSuggestionOut(BaseModel):
    recently_reviewed: List[ProductCardOut] = Field(default_factory=list)
    similar_products: List[ProductCardOut] = Field(default_factory=list)


class SubmitReviewIn(BaseModel):
    product_id: str = Field(min_length=1, max_length=128)
    product_name: Optional[str] = Field(default=None, max_length=255)
    rating: int = Field(ge=1, le=5)
    review_text: str = Field(min_length=3)
    review_title: Optional[str] = Field(default=None, max_length=255)
    pros: Optional[str] = None
    cons: Optional[str] = None
    recommendation: Optional[bool] = None


class SubmitReviewOut(BaseModel):
    review_id: int
    product_id: str
    linked_review_id: Optional[int] = None
