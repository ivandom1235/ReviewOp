# proto/backend/routes/infer.py
from __future__ import annotations

import io
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from core.db import get_db
from models.schemas import JobCreateOut
from models.tables import Job
from services.seq2seq_infer import Seq2SeqEngine
from services.batch_jobs import process_csv_sync


router = APIRouter(prefix="/infer", tags=["infer"])


@router.post("/csv", response_model=JobCreateOut)
async def infer_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"CSV parse failed: {ex}")

    job = Job(status="queued", total=0, processed=0, failed=0)
    db.add(job)
    db.commit()

    from app import app as fastapi_app
    engine: Seq2SeqEngine = fastapi_app.state.seq2seq_engine

    process_csv_sync(db=db, engine=engine, job=job, df=df)

    return JobCreateOut(job_id=job.id, status=job.status, total=job.total)