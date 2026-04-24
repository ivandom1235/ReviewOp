# proto/backend/services/hybrid_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sqlalchemy import delete
from sqlalchemy.orm import Session

from models.tables import AbstainedPrediction, EvidenceSpan, NovelCandidate, Prediction
from services.hybrid_merge import merge_predictions
from services.review_pipeline import run_single_review_pipeline, run_single_review_pipeline_for_existing_review, split_selective_states

PredictionLike = Dict[str, Any]


def _first_evidence_span(row: PredictionLike) -> dict[str, Any]:
    spans = row.get("evidence_spans") or []
    if not spans:
        return {}
    return spans[0] if isinstance(spans[0], dict) else {}


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
        aspect = str(row.get("aspect_raw") or row.get("aspect") or "").strip()
        cluster = str(row.get("aspect_cluster") or row.get("aspect") or row.get("aspect_raw") or "").strip()
        prediction = Prediction(
            aspect_raw=aspect,
            aspect_cluster=cluster,
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


def _persist_selective_states(db: Session, review_obj, selective_states: Dict[str, Any]) -> None:
    db.execute(delete(AbstainedPrediction).where(AbstainedPrediction.review_id == review_obj.id))
    db.execute(delete(NovelCandidate).where(NovelCandidate.review_id == review_obj.id))

    for row in selective_states.get("abstained_predictions", []) or []:
        db.add(
            AbstainedPrediction(
                review_id=review_obj.id,
                reason=str(row.get("reason") or "low_selective_confidence"),
                confidence=float(row.get("confidence", 0.0)),
                ambiguity_score=float(row.get("ambiguity_score", 0.0)),
            )
        )

    seen_novel: set[str] = set()
    for row in selective_states.get("novel_candidates", []) or []:
        aspect = str(row.get("aspect") or row.get("aspect_raw") or row.get("aspect_cluster") or "").strip()
        if not aspect:
            continue
        novelty_score = float(row.get("novelty_score", 0.0))
        confidence = row.get("confidence", None)
        first_span = _first_evidence_span(row)
        key = f"{aspect}|{novelty_score:.6f}|{str(first_span.get('snippet') or '')}"
        if key in seen_novel:
            continue
        seen_novel.add(key)
        db.add(
            NovelCandidate(
                review_id=review_obj.id,
                aspect=aspect,
                novelty_score=novelty_score,
                confidence=float(confidence) if confidence is not None else None,
                evidence=str(first_span.get("snippet") or "") or None,
                evidence_start=int(first_span.get("start_char")) if first_span.get("start_char") is not None else None,
                evidence_end=int(first_span.get("end_char")) if first_span.get("end_char") is not None else None,
            )
        )

    db.flush()


def run_single_review_hybrid_pipeline(
    db: Session,
    *,
    explicit_engine,
    implicit_client,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
    review=None,
    replace_existing: bool = False,
) -> Tuple[Any, List[PredictionLike], List[PredictionLike], List[PredictionLike]]:
    # Step A: run your existing explicit pipeline and persist review/predictions as before
    if review is None:
        review_obj = run_single_review_pipeline(
            db,
            engine=explicit_engine,
            text=text,
            domain=domain,
            product_id=product_id,
        )
    else:
        review_obj = run_single_review_pipeline_for_existing_review(
            db,
            review=review,
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

    # Step C: V6 selective routing filter for implicit outputs
    selective_states = split_selective_states(implicit_predictions)
    accepted_implicit = list(selective_states.get("accepted_predictions", []))

    # Step D: merge explicit + accepted implicit only
    merged_predictions = merge_predictions(
        explicit_predictions=explicit_predictions,
        implicit_predictions=accepted_implicit,
    )

    # V6-only: verifier no longer overrides selective states.
    _persist_final_predictions(db, review_obj, merged_predictions)
    _persist_selective_states(db, review_obj, selective_states)

    return review_obj, explicit_predictions, implicit_predictions, merged_predictions
