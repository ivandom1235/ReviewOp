from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.orm import Session

from models.tables import EvidenceSpan, Prediction, Review
from services.evidence import find_evidence_for_aspect
from services.kg_build import KGBuilder, KGConfig
from services.open_aspect import extract_open_aspects


def _safe_extract_aspects(text: str, max_aspects: int = 8) -> list[str]:
    try:
        aspects = extract_open_aspects(text, max_aspects=max_aspects)
        if aspects:
            return aspects
    except Exception:
        pass
    return ["general"]


def run_single_review_pipeline(
    db: Session,
    *,
    engine,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
    review: Review | None = None,
    replace_existing: bool = False,
) -> Review:
    clean_text = (text or "").strip()
    if review is None:
        review = Review(text=clean_text, domain=domain, product_id=product_id)
        db.add(review)
        db.flush()
    else:
        review.text = clean_text
        review.domain = domain
        review.product_id = product_id
        if replace_existing:
            old_preds = db.query(Prediction).filter(Prediction.review_id == review.id).all()
            old_pred_ids = [pred.id for pred in old_preds if pred.id is not None]
            if old_pred_ids:
                db.execute(delete(EvidenceSpan).where(EvidenceSpan.prediction_id.in_(old_pred_ids)))
            for pred in old_preds:
                db.delete(pred)
            if old_preds:
                db.flush()
            db.expire(review, ["predictions"])

    aspects = _safe_extract_aspects(clean_text, max_aspects=8)

    for aspect_raw in aspects:
        start_char, end_char, snippet = find_evidence_for_aspect(clean_text, aspect_raw)
        sent, conf = engine.classify_sentiment_with_confidence(snippet, aspect_raw)

        pred = Prediction(
            aspect_raw=aspect_raw,
            aspect_cluster=aspect_raw,
            sentiment=sent,
            confidence=float(conf),
            rationale=None,
        )
        pred.review = review
        pred.evidence_spans.append(
            EvidenceSpan(
                start_char=start_char,
                end_char=end_char,
                snippet=snippet,
            )
        )
        db.add(pred)

    db.flush()
    return review


def refresh_corpus_graph(db: Session, domain: str | None = None) -> dict:
    builder = KGBuilder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return builder.rebuild(db=db, domain=domain, cfg=KGConfig())
