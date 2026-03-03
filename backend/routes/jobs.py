# proto/backend/routes/jobs.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.db import get_db
from models.schemas import JobStatusOut
from models.tables import Job

from services.kg_build import KGBuilder, KGConfig

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusOut)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusOut(
        job_id=job.id,
        status=job.status,
        total=job.total,
        processed=job.processed,
        failed=job.failed,
        error=job.error,
    )


@router.post("/kg_rebuild")
def kg_rebuild(domain: str | None = None, db: Session = Depends(get_db)):
    """
    Rebuilds:
    - aspect graph (similarity + cooccurrence)
    - aspect_cluster normalization
    - aspect_nodes stats (df/idf/centrality)
    - per-review weights + overall sentiment
    """
    builder = KGBuilder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return builder.rebuild(db=db, domain=domain, cfg=KGConfig())