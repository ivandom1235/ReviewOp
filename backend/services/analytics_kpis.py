from __future__ import annotations

from typing import Optional

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from models.tables import Prediction, Review
from services.analytics_common import parse_dt
from services.analytics_aspects import aspect_sentiment_distribution, emerging_aspects


def overview(db: Session, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]) -> dict:
    f = parse_dt(dt_from)
    t = parse_dt(dt_to)

    q_reviews = db.query(func.count(Review.id))
    q_preds = db.query(func.count(Prediction.id)).join(Review, Review.id == Prediction.review_id)
    q_unique = db.query(func.count(func.distinct(Prediction.aspect_raw))).join(Review, Review.id == Prediction.review_id)
    q_avg_conf = db.query(func.avg(Prediction.confidence)).join(Review, Review.id == Prediction.review_id)

    if domain:
        q_reviews = q_reviews.filter(Review.domain == domain)
        q_preds = q_preds.filter(Review.domain == domain)
        q_unique = q_unique.filter(Review.domain == domain)
        q_avg_conf = q_avg_conf.filter(Review.domain == domain)

    if f:
        q_reviews = q_reviews.filter(Review.created_at >= f)
        q_preds = q_preds.filter(Review.created_at >= f)
        q_unique = q_unique.filter(Review.created_at >= f)
        q_avg_conf = q_avg_conf.filter(Review.created_at >= f)

    if t:
        q_reviews = q_reviews.filter(Review.created_at <= t)
        q_preds = q_preds.filter(Review.created_at <= t)
        q_unique = q_unique.filter(Review.created_at <= t)
        q_avg_conf = q_avg_conf.filter(Review.created_at <= t)

    total_reviews = int(q_reviews.scalar() or 0)
    total_mentions = int(q_preds.scalar() or 0)
    unique_aspects = int(q_unique.scalar() or 0)
    avg_conf = float(q_avg_conf.scalar() or 0.0)

    q_sent = db.query(
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("pos"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neu"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q_sent = q_sent.filter(Review.domain == domain)
    if f:
        q_sent = q_sent.filter(Review.created_at >= f)
    if t:
        q_sent = q_sent.filter(Review.created_at <= t)

    row = q_sent.first()
    sentiment_counts = {
        "positive": int(row.pos or 0),
        "neutral": int(row.neu or 0),
        "negative": int(row.neg or 0),
    }

    return {
        "total_reviews": total_reviews,
        "total_aspect_mentions": total_mentions,
        "unique_aspects_raw": unique_aspects,
        "avg_confidence": round(avg_conf, 4),
        "sentiment_counts": sentiment_counts,
    }


def dashboard_kpis(db: Session, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]) -> dict:
    base = overview(db, dt_from, dt_to, domain)
    total_mentions = int(base["total_aspect_mentions"] or 0)
    negative_count = int(base["sentiment_counts"]["negative"] or 0)
    negative_sentiment_pct = round((negative_count / total_mentions) * 100, 2) if total_mentions else 0.0

    rows = aspect_sentiment_distribution(db, 100, dt_from, dt_to, domain)
    most_negative_aspect = None
    if rows:
        ranked = sorted(rows, key=lambda r: (-(r["negative"] / max((r["positive"] + r["neutral"] + r["negative"]), 1)), -r["negative"], r["aspect"]))
        most_negative_aspect = ranked[0]["aspect"]

    emerging = emerging_aspects(db, interval="day", lookback_buckets=7, domain=domain)
    return {
        "total_reviews": base["total_reviews"],
        "total_aspects": base["total_aspect_mentions"],
        "most_negative_aspect": most_negative_aspect,
        "negative_sentiment_pct": negative_sentiment_pct,
        "emerging_issues_count": len(emerging),
    }
