from __future__ import annotations

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from models.schemas import InferReviewOut
from services.hybrid_pipeline import run_single_review_hybrid_pipeline
from services.review_pipeline import _refresh_corpus_graph_task
from services.responses import ContractMapper


def infer_review(
    db: Session,
    *,
    explicit_engine,
    implicit_client,
    text: str,
    domain: str | None = None,
    product_id: str | None = None,
    persist: bool = True,
    background_tasks: BackgroundTasks | None = None,
) -> InferReviewOut:
    review_obj, _, implicit_predictions, final_predictions = run_single_review_hybrid_pipeline(
        db,
        explicit_engine=explicit_engine,
        implicit_client=implicit_client,
        text=text,
        domain=domain,
        product_id=product_id,
    )

    response = ContractMapper().to_infer_review_out(review_obj, final_predictions, implicit_predictions)
    if persist:
        db.commit()
        db.refresh(review_obj)
        if background_tasks:
            background_tasks.add_task(_refresh_corpus_graph_task, review_obj.domain or None)
        return response

    db.rollback()
    response.review_id = 0
    return response
