# proto/backend/services/hybrid_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sqlalchemy import delete
from sqlalchemy.orm import Session

from models.tables import EvidenceSpan, Prediction
from services.hybrid_merge import merge_predictions
from services.review_pipeline import run_single_review_pipeline


PredictionLike = Dict[str, Any]


def _prediction_row_to_dict(pred) -> PredictionLike:
    spans = []
    for ev in getattr(pred, "evidence_spans", []) or []:
        spans.append(
            {
                "start_char": int(ev.start_char),
                "end_char": int(ev.end_char),
                "snippet": ev.snippet,
            }
        )

    return {
        "aspect_raw": pred.aspect_raw,
        "aspect_cluster": pred.aspect_cluster,
        "sentiment": pred.sentiment,
        "confidence": float(pred.confidence),
        "evidence_spans": spans,
        "rationale": getattr(pred, "rationale", "") or "",
        "source": "explicit",
    }


def _persist_final_predictions(db: Session, review_obj, final_predictions: List[PredictionLike]) -> None:
    existing_predictions = db.query(Prediction).filter(Prediction.review_id == review_obj.id).all()
    existing_prediction_ids = [pred.id for pred in existing_predictions if pred.id is not None]
    if existing_prediction_ids:
        db.execute(delete(EvidenceSpan).where(EvidenceSpan.prediction_id.in_(existing_prediction_ids)))
    for pred in existing_predictions:
        db.delete(pred)
    if existing_predictions:
        db.flush()
    db.expire(review_obj, ["predictions"])

    for row in final_predictions:
        prediction = Prediction(
            aspect_raw=str(row.get("aspect_raw") or "").strip(),
            aspect_cluster=str(row.get("aspect_cluster") or row.get("aspect_raw") or "").strip(),
            sentiment=str(row.get("sentiment") or "neutral").strip().lower(),
            confidence=float(row.get("confidence", 0.0)),
            rationale=str(row.get("rationale") or "").strip() or None,
        )
        prediction.review = review_obj

        for ev in row.get("evidence_spans", []) or []:
            prediction.evidence_spans.append(
                EvidenceSpan(
                    start_char=int(ev.get("start_char", 0)),
                    end_char=int(ev.get("end_char", 0)),
                    snippet=str(ev.get("snippet", "")),
                )
            )

        db.add(prediction)

    db.flush()


def run_single_review_hybrid_pipeline(
    db: Session,
    *,
    explicit_engine,
    implicit_client,
    llm_verifier,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
) -> Tuple[Any, List[PredictionLike], List[PredictionLike], List[PredictionLike]]:
    # Step A: run your existing explicit pipeline and persist review/predictions as before
    review_obj = run_single_review_pipeline(
        db,
        engine=explicit_engine,
        text=text,
        domain=domain,
        product_id=product_id,
    )

    explicit_predictions: List[PredictionLike] = [
        _prediction_row_to_dict(pred) for pred in getattr(review_obj, "predictions", []) or []
    ]

    # Step B: implicit candidates from protonet
    implicit_predictions = implicit_client.predict(
        review_text=text,
        domain=domain,
    )
    print("EXPLICIT:", explicit_predictions)
    print("IMPLICIT:", implicit_predictions)

    # Step C: merge explicit + implicit
    merged_predictions = merge_predictions(
        explicit_predictions=explicit_predictions,
        implicit_predictions=implicit_predictions,
    )

    # Step D: verifier
    verified_predictions = llm_verifier.verify(
        review_text=text,
        explicit_predictions=explicit_predictions,
        implicit_predictions=implicit_predictions,
        merged_predictions=merged_predictions,
    )

    _persist_final_predictions(db, review_obj, verified_predictions)

    return review_obj, explicit_predictions, implicit_predictions, verified_predictions
