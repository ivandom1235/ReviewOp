# proto/backend/routes/analytics.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import OverviewOut, TopAspectOut, AspectSentimentDistOut, TrendPointOut
from services.analytics import (
    overview,
    top_aspects,
    aspect_sentiment_distribution,
    trends,
    dashboard_kpis,
    aspect_leaderboard,
    evidence_drilldown,
    aspect_trends,
    emerging_aspects,
    aspect_detail,
    alerts,
    impact_matrix,
    segment_drilldown,
    weekly_summary,
)

# add imports
from models.schemas import CentralityOut, CommunityOut, EdgeOut
from models.schemas import (
    DashboardKpiOut,
    AspectLeaderboardRowOut,
    EvidenceRowOut,
    AspectTrendPointOut,
    AspectDetailOut,
    AlertOut,
    ImpactMatrixRowOut,
    SegmentDrilldownOut,
    WeeklySummaryOut,
)
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


@router.get("/dashboard_kpis", response_model=DashboardKpiOut)
def analytics_dashboard_kpis(
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return dashboard_kpis(db, dt_from, dt_to, domain)


@router.get("/aspect_leaderboard", response_model=list[AspectLeaderboardRowOut])
def analytics_aspect_leaderboard(
    limit: int = 25,
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return aspect_leaderboard(db, limit=limit, domain=domain)


@router.get("/evidence", response_model=list[EvidenceRowOut])
def analytics_evidence(
    aspect: str | None = None,
    sentiment: str | None = None,
    limit: int = 50,
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return evidence_drilldown(db, aspect=aspect, sentiment=sentiment, limit=limit, domain=domain)


@router.get("/aspect_trends", response_model=list[AspectTrendPointOut])
def analytics_aspect_trends(
    interval: str = "day",
    domain: str | None = None,
    limit: int = 500,
    db: Session = Depends(get_db),
):
    return aspect_trends(db, interval=interval, domain=domain, limit=limit)


@router.get("/emerging_aspects")
def analytics_emerging_aspects(
    interval: str = "day",
    lookback_buckets: int = 7,
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return emerging_aspects(db, interval=interval, lookback_buckets=lookback_buckets, domain=domain)


@router.get("/aspect_detail/{aspect}", response_model=AspectDetailOut)
def analytics_aspect_detail(
    aspect: str,
    interval: str = "day",
    domain: str | None = None,
    db: Session = Depends(get_db),
):
    return aspect_detail(db, aspect=aspect, interval=interval, domain=domain)


@router.get("/alerts", response_model=list[AlertOut])
def analytics_alerts(domain: str | None = None, db: Session = Depends(get_db)):
    return alerts(db, domain=domain)


@router.get("/impact_matrix", response_model=list[ImpactMatrixRowOut])
def analytics_impact_matrix(limit: int = 20, domain: str | None = None, db: Session = Depends(get_db)):
    return impact_matrix(db, domain=domain, limit=limit)


@router.get("/segments", response_model=list[SegmentDrilldownOut])
def analytics_segments(limit: int = 20, domain: str | None = None, db: Session = Depends(get_db)):
    return segment_drilldown(db, domain=domain, limit=limit)


@router.get("/weekly_summary", response_model=WeeklySummaryOut)
def analytics_weekly_summary(domain: str | None = None, db: Session = Depends(get_db)):
    return weekly_summary(db, domain=domain)

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
