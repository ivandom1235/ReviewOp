"""Schema and file format detection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


TEXT_CANDIDATES = ["review_text", "review", "text", "content", "comment", "body", "sentence"]
ASPECT_CANDIDATES = ["aspect", "aspects", "category", "aspect_term", "implicit_aspect"]
SENTIMENT_CANDIDATES = ["sentiment", "polarity", "label", "class"]
ID_CANDIDATES = ["id", "review_id", "uid", "example_id"]
SPLIT_CANDIDATES = ["split", "set", "partition"]
DOMAIN_CANDIDATES = ["domain", "dataset", "vertical", "category_domain"]
EVIDENCE_CANDIDATES = ["evidence", "evidence_sentence", "sentence", "snippet"]
SPAN_FROM_CANDIDATES = ["from", "start", "begin", "offset_start", "char_from"]
SPAN_TO_CANDIDATES = ["to", "end", "offset_end", "char_to"]


@dataclass
class DetectedSchema:
    file_path: str
    file_type: str
    columns: List[str]
    text_col: str | None
    aspect_cols: List[str]
    sentiment_col: str | None
    id_col: str | None
    split_col: str | None
    domain_col: str | None
    evidence_col: str | None
    span_from_col: str | None
    span_to_col: str | None


def _match_column(columns: List[str], candidates: List[str]) -> str | None:
    lc = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    for col in columns:
        low = col.lower()
        if any(cand in low for cand in candidates):
            return col
    return None


def load_file_rows(path: Path) -> Tuple[List[Dict], str, List[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    elif suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        cols = sorted({k for r in rows for k in r.keys()})
        return rows, "jsonl", cols
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                rows = payload["data"]
            elif isinstance(payload.get("records"), list):
                rows = payload["records"]
            else:
                rows = [payload]
        else:
            rows = []
        cols = sorted({k for r in rows if isinstance(r, dict) for k in r.keys()})
        return rows, "json", cols
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df = df.fillna("")
    rows = df.to_dict(orient="records")
    return rows, suffix.lstrip("."), list(df.columns)


def detect_schema(path: Path, columns: List[str], file_type: str) -> DetectedSchema:
    text_col = _match_column(columns, TEXT_CANDIDATES)
    aspect_col = _match_column(columns, ASPECT_CANDIDATES)
    sentiment_col = _match_column(columns, SENTIMENT_CANDIDATES)
    id_col = _match_column(columns, ID_CANDIDATES)
    split_col = _match_column(columns, SPLIT_CANDIDATES)
    domain_col = _match_column(columns, DOMAIN_CANDIDATES)
    evidence_col = _match_column(columns, EVIDENCE_CANDIDATES)
    span_from_col = _match_column(columns, SPAN_FROM_CANDIDATES)
    span_to_col = _match_column(columns, SPAN_TO_CANDIDATES)

    return DetectedSchema(
        file_path=str(path),
        file_type=file_type,
        columns=columns,
        text_col=text_col,
        aspect_cols=[aspect_col] if aspect_col else [],
        sentiment_col=sentiment_col,
        id_col=id_col,
        split_col=split_col,
        domain_col=domain_col,
        evidence_col=evidence_col,
        span_from_col=span_from_col,
        span_to_col=span_to_col,
    )
