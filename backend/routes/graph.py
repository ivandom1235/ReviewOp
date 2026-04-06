from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import GraphResponseOut
from services.graph_builders import build_batch_aspect_graph, build_single_review_graph


router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/review/{review_id}", response_model=GraphResponseOut)
def review_graph(review_id: int, db: Session = Depends(get_db)):
    payload = build_single_review_graph(db, review_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return payload


@router.get("/aspects", response_model=GraphResponseOut)
def batch_graph(
    domain: str | None = None,
    product_id: str | None = None,
    dt_from: str | None = Query(default=None, alias="from"),
    dt_to: str | None = Query(default=None, alias="to"),
    min_edge_weight: int = 1,
    graph_mode: str = Query(default="accepted", pattern="^(accepted|novel_side)$"),
    db: Session = Depends(get_db),
):
    return build_batch_aspect_graph(
        db=db,
        domain=domain,
        product_id=product_id,
        dt_from=dt_from,
        dt_to=dt_to,
        min_edge_weight=min_edge_weight,
        graph_mode=graph_mode,
    )
