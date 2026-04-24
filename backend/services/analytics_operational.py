from __future__ import annotations

from sqlalchemy.orm import Session

from models.tables import AbstainedPrediction, NovelCandidate, Review


def needs_review_queue(db: Session, *, limit: int = 100) -> list[dict]:
    rows = (
        db.query(AbstainedPrediction, Review)
        .join(Review, Review.id == AbstainedPrediction.review_id)
        .order_by(AbstainedPrediction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": item.id,
            "review_id": review.id,
            "review_text": review.text,
            "domain": review.domain,
            "product_id": review.product_id,
            "reason": item.reason,
            "confidence": float(item.confidence),
            "ambiguity_score": float(item.ambiguity_score),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item, review in rows
    ]


def novel_candidates_queue(db: Session, *, limit: int = 100) -> list[dict]:
    rows = (
        db.query(NovelCandidate, Review)
        .join(Review, Review.id == NovelCandidate.review_id)
        .order_by(NovelCandidate.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": item.id,
            "review_id": review.id,
            "review_text": review.text,
            "domain": review.domain,
            "product_id": review.product_id,
            "aspect": item.aspect,
            "novelty_score": float(item.novelty_score),
            "confidence": float(item.confidence) if item.confidence is not None else None,
            "evidence": item.evidence,
            "evidence_start": item.evidence_start,
            "evidence_end": item.evidence_end,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item, review in rows
    ]
