# proto/backend/models/schemas.py
from typing import List, Optional
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

    # NEW:
    aspect_weight: Optional[float] = None
    aspect_score: Optional[float] = None

    evidence_spans: List[EvidenceSpanOut] = Field(default_factory=list)
    rationale: Optional[str] = None


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