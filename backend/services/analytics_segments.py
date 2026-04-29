from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import case, func, text
from sqlalchemy.orm import Session

from models.tables import Prediction, Review
from services.analytics_aspects import aspect_leaderboard, emerging_aspects


def impact_matrix(db: Session, domain: Optional[str] = None, limit: int = 20) -> list[dict]:
    rows = aspect_leaderboard(db, limit=max(limit, 30), domain=domain)
    out = []
    for row in rows:
        negative_rate = float(row["negative_pct"]) / 100.0
        growth_factor = 1 + max(0.0, float(row["change_7d_vs_prev_7d"]) / 100.0)
        priority = float(row["frequency"]) * negative_rate * growth_factor
        tier = "high" if priority >= 8 else "medium" if priority >= 3 else "low"
        out.append({"aspect": row["aspect"], "volume": int(row["frequency"]), "negative_rate": round(negative_rate, 4), "growth_pct": float(row["change_7d_vs_prev_7d"]), "priority_score": round(priority, 3), "action_tier": tier})
    out.sort(key=lambda item: (-item["priority_score"], item["aspect"]))
    return out[: max(1, limit)]


def segment_drilldown(db: Session, domain: Optional[str] = None, limit: int = 20) -> list[dict]:
    out = []
    domain_q = db.query(Review.domain.label("segment"), func.count(func.distinct(Review.id)).label("reviews"), func.count(Prediction.id).label("mentions"), func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg")).join(Prediction, Prediction.review_id == Review.id)
    if domain:
        domain_q = domain_q.filter(Review.domain == domain)
    domain_q = domain_q.group_by(Review.domain)
    domain_rows = domain_q.all()
    domain_top_neg_q = db.query(Review.domain.label("segment"), Prediction.aspect_raw.label("aspect"), func.count(Prediction.id).label("c")).join(Prediction, Prediction.review_id == Review.id).filter(Prediction.sentiment == "negative")
    if domain:
        domain_top_neg_q = domain_top_neg_q.filter(Review.domain == domain)
    domain_top_neg_rows = domain_top_neg_q.group_by(Review.domain, Prediction.aspect_raw).all()
    domain_top_negative: dict[str, tuple[str, int]] = {}
    for row in domain_top_neg_rows:
        key = row.segment or "unknown"
        current = domain_top_negative.get(key)
        count = int(row.c or 0)
        aspect = str(row.aspect)
        if current is None or count > current[1] or (count == current[1] and aspect < current[0]):
            domain_top_negative[key] = (aspect, count)
    for row in domain_rows:
        mentions = int(row.mentions or 0)
        neg = int(row.neg or 0)
        segment_key = row.segment or "unknown"
        top_neg = domain_top_negative.get(segment_key)
        out.append({"segment_type": "domain", "segment_value": segment_key, "review_count": int(row.reviews or 0), "mention_count": mentions, "negative_pct": round((neg / mentions) * 100, 2) if mentions else 0.0, "top_negative_aspect": top_neg[0] if top_neg else None})
    product_q = db.query(Review.product_id.label("segment"), func.count(func.distinct(Review.id)).label("reviews"), func.count(Prediction.id).label("mentions"), func.sum(case((Prediction.sentiment == "negative", 1), else_=0)).label("neg")).join(Prediction, Prediction.review_id == Review.id).filter(Review.product_id.isnot(None))
    if domain:
        product_q = product_q.filter(Review.domain == domain)
    product_q = product_q.group_by(Review.product_id).order_by(text("mentions DESC")).limit(limit)
    for row in product_q.all():
        mentions = int(row.mentions or 0)
        neg = int(row.neg or 0)
        out.append({"segment_type": "product_id", "segment_value": row.segment or "unknown", "review_count": int(row.reviews or 0), "mention_count": mentions, "negative_pct": round((neg / mentions) * 100, 2) if mentions else 0.0, "top_negative_aspect": None})
    out.sort(key=lambda item: (-item["negative_pct"], -item["mention_count"], item["segment_type"], item["segment_value"]))
    return out[: max(1, limit)]


def weekly_summary(db: Session, domain: Optional[str] = None) -> dict:
    now = datetime.utcnow()
    this_week_start = now - timedelta(days=7)
    prev_week_start = now - timedelta(days=14)
    current = aspect_leaderboard(db, limit=12, domain=domain)
    impact = impact_matrix(db, domain=domain, limit=3)
    emerging = emerging_aspects(db, interval="day", lookback_buckets=7, domain=domain)
    biggest = max(current, key=lambda row: row["change_7d_vs_prev_7d"], default=None)
    recommendations = [f"Prioritize remediation for {row['aspect']} (score {row['priority_score']})" for row in impact]
    return {"period_label": f"{this_week_start.date()} to {now.date()} vs {prev_week_start.date()} to {this_week_start.date()}", "top_drivers": [row["aspect"] for row in impact], "biggest_increase_aspect": biggest["aspect"] if biggest else None, "biggest_increase_pct": float(biggest["change_7d_vs_prev_7d"]) if biggest else 0.0, "emerging_count": len(emerging), "action_recommendations": recommendations}
