# proto/backend/routes/analytics.py
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import (
    AdminExportOut,
    AlertOut,
    AspectDetailOut,
    AspectLeaderboardRowOut,
    AspectSentimentDistOut,
    AspectTrendPointOut,
    CentralityOut,
    CommunityOut,
    DashboardKpiOut,
    EdgeOut,
    EvidenceRowOut,
    ImpactMatrixRowOut,
    OverviewOut,
    SegmentDrilldownOut,
    TopAspectOut,
    TrendPointOut,
    UserReviewListOut,
    UserReviewSummaryOut,
    WeeklySummaryOut,
)
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
    clear_alert,
    export_payload,
    export_pdf_bytes,
    user_reviews_list,
    user_reviews_summary,
    needs_review_queue,
    novel_candidates_queue,
)
from routes.user_portal import require_admin

from services.kg_analytics import centrality_leaderboard, edges as kg_edges, communities as kg_communities

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview", response_model=OverviewOut)
def analytics_overview(
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return overview(db, dt_from, dt_to, domain)


@router.get("/top_aspects", response_model=list[TopAspectOut])
def analytics_top_aspects(
    limit: int = 10,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return top_aspects(db, limit, dt_from, dt_to, domain)


@router.get("/aspect_sentiment_distribution", response_model=list[AspectSentimentDistOut])
def analytics_aspect_sentiment_distribution(
    limit: int = 10,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    domain: str | None = None,
    _: None = Depends(require_admin),
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
    _: None = Depends(require_admin),
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
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return dashboard_kpis(db, dt_from, dt_to, domain)


@router.get("/aspect_leaderboard", response_model=list[AspectLeaderboardRowOut])
def analytics_aspect_leaderboard(
    limit: int = 25,
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return aspect_leaderboard(db, limit=limit, domain=domain)


@router.get("/evidence", response_model=list[EvidenceRowOut])
def analytics_evidence(
    aspect: str | None = None,
    sentiment: str | None = None,
    limit: int = 50,
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return evidence_drilldown(db, aspect=aspect, sentiment=sentiment, limit=limit, domain=domain)


@router.get("/aspect_trends", response_model=list[AspectTrendPointOut])
def analytics_aspect_trends(
    interval: str = "day",
    domain: str | None = None,
    limit: int = 500,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    interval = interval.lower().strip()
    if interval not in {"day", "week"}:
        interval = "day"
    return aspect_trends(db, interval=interval, domain=domain, limit=limit)


@router.get("/emerging_aspects")
def analytics_emerging_aspects(
    interval: str = "day",
    lookback_buckets: int = 7,
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    interval = interval.lower().strip()
    if interval not in {"day", "week"}:
        interval = "day"
    return emerging_aspects(db, interval=interval, lookback_buckets=lookback_buckets, domain=domain)


@router.get("/aspect_detail/{aspect}", response_model=AspectDetailOut)
def analytics_aspect_detail(
    aspect: str,
    interval: str = "day",
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return aspect_detail(db, aspect=aspect, interval=interval, domain=domain)


@router.get("/alerts", response_model=list[AlertOut])
def analytics_alerts(domain: str | None = None, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return alerts(db, domain=domain)


@router.get("/needs_review")
def analytics_needs_review(limit: int = 100, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return needs_review_queue(db, limit=limit)


@router.get("/novel_candidates")
def analytics_novel_candidates(limit: int = 100, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return novel_candidates_queue(db, limit=limit)


@router.delete("/alerts/{alert_id}")
def analytics_clear_alert(alert_id: int, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    ok = clear_alert(db, alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail="alert not found")
    return {"ok": True}


@router.get("/impact_matrix", response_model=list[ImpactMatrixRowOut])
def analytics_impact_matrix(limit: int = 20, domain: str | None = None, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return impact_matrix(db, domain=domain, limit=limit)


@router.get("/segments", response_model=list[SegmentDrilldownOut])
def analytics_segments(limit: int = 20, domain: str | None = None, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return segment_drilldown(db, domain=domain, limit=limit)


@router.get("/weekly_summary", response_model=WeeklySummaryOut)
def analytics_weekly_summary(domain: str | None = None, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return weekly_summary(db, domain=domain)


@router.get("/user_reviews/summary", response_model=UserReviewSummaryOut)
def analytics_user_reviews_summary(domain: str | None = None, _: None = Depends(require_admin), db: Session = Depends(get_db)):
    return user_reviews_summary(db, domain=domain)


@router.get("/user_reviews/list", response_model=UserReviewListOut)
def analytics_user_reviews_list(
    domain: str | None = None,
    product_id: str | None = None,
    username: str | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    limit: int = 50,
    offset: int = 0,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return user_reviews_list(
        db,
        domain=domain,
        product_id=product_id,
        username=username,
        min_rating=min_rating,
        max_rating=max_rating,
        limit=limit,
        offset=offset,
    )


@router.get("/export/json", response_model=AdminExportOut)
def analytics_export_json(
    domain: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return export_payload(db, domain=domain, limit=limit, offset=offset)


@router.get("/export/pdf")
def analytics_export_pdf(
    domain: str | None = None,
    limit: int = 100,
    offset: int = 0,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    pdf_bytes = export_pdf_bytes(db, domain=domain, limit=limit, offset=offset)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=reviewop-admin-export.pdf"},
    )

@router.get("/kg/centrality", response_model=list[CentralityOut])
def analytics_kg_centrality(
    limit: int = 20,
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return centrality_leaderboard(db, limit=limit, domain=domain)


@router.get("/kg/edges", response_model=list[EdgeOut])
def analytics_kg_edges(
    limit: int = 200,
    edge_type: str | None = None,  # similarity|cooccurrence
    domain: str | None = None,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return kg_edges(db, limit=limit, domain=domain, edge_type=edge_type)


@router.get("/kg/communities", response_model=list[CommunityOut])
def analytics_kg_communities(
    domain: str | None = None,
    edge_type: str = "cooccurrence",
    min_weight: float = 2.0,
    _: None = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return kg_communities(db, domain=domain, edge_type=edge_type, min_weight=min_weight)
