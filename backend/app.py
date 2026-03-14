# proto/backend/app.py
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text, inspect
from sqlalchemy.orm import Session

from core.db import engine, Base, get_db, SessionLocal
from core.config import settings
from models.tables import Review, Prediction, EvidenceSpan
from models.schemas import InferReviewIn, InferReviewOut, PredictionOut, EvidenceSpanOut
from services.seq2seq_infer import Seq2SeqEngine
from services.evidence import find_evidence_for_aspect
from services.open_aspect import extract_open_aspects

from routes.analytics import router as analytics_router
from routes.graph import router as graph_router
from routes.jobs import router as jobs_router
from routes.infer import router as infer_router
from routes.user_portal import router as user_portal_router, seed_default_accounts


app = FastAPI(title="Proto ReviewOps MVP (Phase 1-2)", version="0.3.0")


def _safe_extract_aspects(text: str, max_aspects: int = 8) -> list[str]:
    try:
        aspects = extract_open_aspects(text, max_aspects=max_aspects)
        if aspects:
            return aspects
    except Exception:
        pass
    return ["general"]


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    _apply_schema_patches()
    db = SessionLocal()
    try:
        seed_default_accounts(db)
    finally:
        db.close()
    app.state.seq2seq_engine = Seq2SeqEngine.load()


def _apply_schema_patches() -> None:
    """Apply minimal in-place schema patches for legacy local databases."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        if inspector.has_table("products"):
            product_cols = {c["name"] for c in inspector.get_columns("products")}
            if "cached_average_rating" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_average_rating FLOAT NOT NULL DEFAULT 0.0"))
            if "cached_review_count" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_review_count INTEGER NOT NULL DEFAULT 0"))
            if "cached_latest_review_at" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_latest_review_at DATETIME NULL"))
            if "cached_helpful_count" not in product_cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN cached_helpful_count INTEGER NOT NULL DEFAULT 0"))

        if not inspector.has_table("admin_dismissed_alerts"):
            conn.execute(
                text(
                    """
                    CREATE TABLE admin_dismissed_alerts (
                        id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        type VARCHAR(64) NOT NULL,
                        aspect VARCHAR(255) NOT NULL,
                        message VARCHAR(512) NOT NULL,
                        domain VARCHAR(64) NULL,
                        signature VARCHAR(64) NOT NULL UNIQUE,
                        dismissed_at DATETIME NOT NULL
                    )
                    """
                )
            )


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "ok": True,
            "env": settings.app_env,
            "db": "connected",
            "model": settings.seq2seq_model_name,
        }
    except Exception as ex:
        return {"ok": False, "error": str(ex)}


@app.post("/infer/review", response_model=InferReviewOut, tags=["infer"])
def infer_review(payload: InferReviewIn, db: Session = Depends(get_db)):
    text_in = (payload.text or "").strip()
    if not text_in:
        raise HTTPException(status_code=400, detail="text is required")

    r = Review(text=text_in, domain=payload.domain, product_id=payload.product_id)
    db.add(r)
    db.flush()

    engine_ = app.state.seq2seq_engine

    aspects = _safe_extract_aspects(text_in, max_aspects=8)

    preds_out: list[PredictionOut] = []

    for aspect_raw in aspects:
        # Evidence first: classify sentiment on the evidence snippet (sentence), not full review.
        s, e, snippet = find_evidence_for_aspect(text_in, aspect_raw)

        # Use calibrated confidence (no constant 0.75) via likelihood scoring.
        sent, conf = engine_.classify_sentiment_with_confidence(snippet, aspect_raw)

        pred = Prediction(
            review_id=r.id,
            aspect_raw=aspect_raw,
            aspect_cluster=aspect_raw,  # Phase 3: clustering label
            sentiment=sent,
            confidence=float(conf),
            rationale=None,
        )
        db.add(pred)
        db.flush()

        ev = EvidenceSpan(
            prediction_id=pred.id,
            start_char=s,
            end_char=e,
            snippet=snippet,
        )
        db.add(ev)

        preds_out.append(
            PredictionOut(
                aspect_raw=aspect_raw,
                aspect_cluster=aspect_raw,
                sentiment=sent,
                confidence=float(conf),
                evidence_spans=[EvidenceSpanOut(start_char=s, end_char=e, snippet=snippet)],
                rationale=None,
            )
        )

    db.commit()

    return InferReviewOut(
    review_id=r.id,
    domain=r.domain,
    product_id=r.product_id,
    predictions=preds_out,
    overall_sentiment=r.overall_sentiment,
    overall_score=r.overall_score,
    overall_confidence=r.overall_confidence,
)


app.include_router(infer_router)  # contains /infer/csv
app.include_router(jobs_router)
app.include_router(analytics_router)
app.include_router(graph_router)
app.include_router(user_portal_router)
