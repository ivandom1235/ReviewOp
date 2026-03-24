# proto/backend/services/hybrid_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

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

    return review_obj, explicit_predictions, implicit_predictions, verified_predictions
