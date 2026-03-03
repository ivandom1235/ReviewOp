# proto/backend/routes/analytics.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import OverviewOut, TopAspectOut, AspectSentimentDistOut, TrendPointOut
from services.analytics import overview, top_aspects, aspect_sentiment_distribution, trends

# add imports
from models.schemas import CentralityOut, CommunityOut, EdgeOut
from services.kg_analytics import centrality_leaderboard, edges as kg_edges, communities as kg_communities
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=OverviewOut)
def analytics_overview(
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return overview(db, dt_from, dt_to, domain)


@router.get("/top_aspects", response_model=list[TopAspectOut])
def analytics_top_aspects(
    limit: int = 10,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return top_aspects(db, limit, dt_from, dt_to, domain)


@router.get("/aspect_sentiment_distribution", response_model=list[AspectSentimentDistOut])
def analytics_aspect_sentiment_distribution(
    limit: int = 10,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return aspect_sentiment_distribution(db, limit, dt_from, dt_to, domain)


@router.get("/trends", response_model=list[TrendPointOut])
def analytics_trends(
    interval: str = "day",
    aspect: str | None = None,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    interval = interval.lower().strip()
    if interval not in {"day", "week"}:
        interval = "day"
    return trends(db, interval, aspect, dt_from, dt_to, domain)

@router.get("/kg/centrality", response_model=list[CentralityOut])
def analytics_kg_centrality(
    limit: int = 20,
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return centrality_leaderboard(db, limit=limit, domain=domain)


@router.get("/kg/edges", response_model=list[EdgeOut])
def analytics_kg_edges(
    limit: int = 200,
    edge_type: str | None = None,  # similarity|cooccurrence
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return kg_edges(db, limit=limit, domain=domain, edge_type=edge_type)


@router.get("/kg/communities", response_model=list[CommunityOut])
def analytics_kg_communities(
    domain: str | None = None,
    edge_type: str = "cooccurrence",
    min_weight: float = 2.0,
    db: Session = Depends(get_db),
):
    return kg_communities(db, domain=domain, edge_type=edge_type, min_weight=min_weight)