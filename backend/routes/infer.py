# proto/backend/routes/infer.py
from __future__ import annotations

import io
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from pandas.errors import ParserError

from core.db import get_db
from models.schemas import JobCreateOut
from models.tables import Job
from services.seq2seq_infer import Seq2SeqEngine
from services.batch_jobs import process_csv_sync


router = APIRouter(prefix="/infer", tags=["infer"])


def _read_csv_best_effort(content: bytes, encoding: str) -> pd.DataFrame:
    # Normal parser first (fast path).
    try:
        return pd.read_csv(io.BytesIO(content), encoding=encoding)
    except ParserError:
        pass

    # More permissive Python engine.
    try:
        return pd.read_csv(io.BytesIO(content), encoding=encoding, engine="python")
    except ParserError:
        pass

    # Last-resort recovery: treat each non-empty line as one review text row.
    text = content.decode(encoding, errors="strict")
    lines = [line.strip() for line in text.splitlines() if line and line.strip()]
    if not lines:
        raise ValueError("Empty CSV after decoding")

    candidates = {"reviews", "review", "text", "content", "sentence", "comment"}
    header = lines[0].lstrip("\ufeff").strip().strip('"').strip("'").lower()
    data_lines = lines[1:] if header in candidates else lines
    reviews = [line.strip().strip('"').replace('""', '"') for line in data_lines if line.strip()]
    if not reviews:
        raise ValueError("No review rows found in CSV")

    return pd.DataFrame({"review": reviews})


@router.post("/csv", response_model=JobCreateOut)
async def infer_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    df = None
    decode_errors: list[str] = []
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            df = _read_csv_best_effort(content, encoding=encoding)
            break
        except UnicodeDecodeError as ex:
            decode_errors.append(f"{encoding}: {ex}")
        except Exception as ex:
            raise HTTPException(status_code=400, detail=f"CSV parse failed: {ex}")

    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parse failed: unable to decode file as utf-8-sig/cp1252/latin1 ({' | '.join(decode_errors)})",
        )

    job = Job(status="queued", total=0, processed=0, failed=0)
    db.add(job)
    db.commit()

    from app import app as fastapi_app
    engine: Seq2SeqEngine = fastapi_app.state.seq2seq_engine

    process_csv_sync(db=db, engine=engine, job=job, df=df)

    return JobCreateOut(job_id=job.id, status=job.status, total=job.total)
