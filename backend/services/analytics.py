# proto/backend/services/analytics.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple

from sqlalchemy import func, case, text
from sqlalchemy.orm import Session

from models.tables import Review, Prediction


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # Accept YYYY-MM-DD or ISO
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None


def overview(db: Session, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]) -> dict:
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q_reviews = db.query(func.count(Review.id))
    q_preds = db.query(func.count(Prediction.id))
    q_unique = db.query(func.count(func.distinct(Prediction.aspect_raw)))
    q_avg_conf = db.query(func.avg(Prediction.confidence))

    if domain:
        q_reviews = q_reviews.filter(Review.domain == domain)

    if f:
        q_reviews = q_reviews.filter(Review.created_at >= f)
        q_preds = q_preds.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)
        q_unique = q_unique.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)
        q_avg_conf = q_avg_conf.join(Review, Review.id == Prediction.review_id).filter(Review.created_at >= f)

    if t:
        q_reviews = q_reviews.filter(Review.created_at <= t)
        q_preds = q_preds.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)
        q_unique = q_unique.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)
        q_avg_conf = q_avg_conf.join(Review, Review.id == Prediction.review_id).filter(Review.created_at <= t)

    total_reviews = int(q_reviews.scalar() or 0)
    total_mentions = int(q_preds.scalar() or 0)
    unique_aspects = int(q_unique.scalar() or 0)
    avg_conf = float(q_avg_conf.scalar() or 0.0)

    # sentiment counts
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


def top_aspects(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.count(Prediction.id).label("count"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(Prediction.aspect_raw).order_by(text("count DESC")).limit(limit)
    return [{"aspect": r.aspect, "count": int(r.count)} for r in q.all()]


def aspect_sentiment_distribution(db: Session, limit: int, dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    q = db.query(
        Prediction.aspect_raw.label("aspect"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("positive"),
        func.sum(case((Prediction.sentiment == "neutral", 1), else_=0)).label("neutral"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("negative"),
        func.count(Prediction.id).label("total"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(Prediction.aspect_raw).order_by(text("total DESC")).limit(limit)

    out = []
    for r in q.all():
        out.append(
            {
                "aspect": r.aspect,
                "positive": int(r.positive or 0),
                "neutral": int(r.neutral or 0),
                "negative": int(r.negative or 0),
            }
        )
    return out


def trends(db: Session, interval: str, aspect: Optional[str], dt_from: Optional[str], dt_to: Optional[str], domain: Optional[str]):
    """
    interval: day|week
    sentiment_score = positive_pct - negative_pct
    """
    f = _parse_dt(dt_from)
    t = _parse_dt(dt_to)

    # MySQL bucket formatting
    if interval == "week":
        bucket_expr = func.date_format(Review.created_at, "%x-W%v")  # ISO week
    else:
        bucket_expr = func.date_format(Review.created_at, "%Y-%m-%d")

    q = db.query(
        bucket_expr.label("bucket"),
        func.count(Prediction.id).label("mentions"),
        func.sum(case((Prediction.sentiment == "positive", 1), else_=0)).label("pos"),
        func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg"),
    ).join(Review, Review.id == Prediction.review_id)

    if domain:
        q = q.filter(Review.domain == domain)
    if aspect:
        q = q.filter(Prediction.aspect_raw == aspect)
    if f:
        q = q.filter(Review.created_at >= f)
    if t:
        q = q.filter(Review.created_at <= t)

    q = q.group_by(bucket_expr).order_by(bucket_expr.asc())

    out = []
    for r in q.all():
        mentions = int(r.mentions or 0)
        pos = int(r.pos or 0)
        neg = int(r.neg or 0)
        neg_pct = (neg / mentions) if mentions > 0 else 0.0
        pos_pct = (pos / mentions) if mentions > 0 else 0.0
        score = pos_pct - neg_pct
        out.append(
            {
                "bucket": str(r.bucket),
                "mentions": mentions,
                "negative_pct": round(neg_pct, 4),
                "sentiment_score": round(score, 4),
            }
        )
    return out