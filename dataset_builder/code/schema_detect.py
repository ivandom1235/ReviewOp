from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_utils import LLMClient

TEXT_KEYS = ["review_text", "review", "text", "content", "comment", "body", "sentence"]
TITLE_KEYS = ["title", "headline", "summary", "subject"]
RATING_KEYS = ["rating", "stars", "score", "overall"]
ID_KEYS = ["review_id", "id", "uid"]
DOMAIN_KEYS = ["domain", "category", "vertical"]
GROUP_KEYS = ["product_id", "item_id", "business_id", "sku"]
SPLIT_KEYS = ["split", "set", "partition"]
SENTIMENT_KEYS = ["sentiment", "polarity", "label"]
DATE_KEYS = ["date", "time", "timestamp", "created_at"]

ASPECT_CANDIDATES = ["aspect", "aspects", "category", "aspect_term", "implicit_aspect"]
EVIDENCE_CANDIDATES = ["evidence", "evidence_sentence", "sentence", "snippet"]
SPAN_FROM_CANDIDATES = ["from", "start", "begin", "offset_start", "char_from"]
SPAN_TO_CANDIDATES = ["to", "end", "offset_end", "char_to"]


@dataclass
class SchemaMapping:
    text_col: Optional[str] = None
    title_col: Optional[str] = None
    rating_col: Optional[str] = None
    id_col: Optional[str] = None
    domain_col: Optional[str] = None
    group_col: Optional[str] = None
    split_col: Optional[str] = None
    sentiment_col: Optional[str] = None
    date_col: Optional[str] = None


@dataclass
class DetectedSchema:
    file_path: str
    file_type: str
    columns: List[str]
    text_col: Optional[str]
    aspect_cols: List[str]
    sentiment_col: Optional[str]
    id_col: Optional[str]
    split_col: Optional[str]
    domain_col: Optional[str]
    evidence_col: Optional[str]
    span_from_col: Optional[str]
    span_to_col: Optional[str]


def _pick(columns: List[str], candidates: List[str]) -> Optional[str]:
    lc = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    for c in columns:
        low = c.lower()
        if any(k in low for k in candidates):
            return c
    return None


def _confidence(col: Optional[str]) -> float:
    return 0.9 if col else 0.0


def _fingerprint(columns: List[str], samples: List[Dict]) -> str:
    basis = json.dumps({"columns": sorted(columns), "sample_keys": sorted({k for r in samples for k in r.keys()})}, sort_keys=True)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _llm_schema_fallback(columns: List[str], samples: List[Dict], llm: Optional[LLMClient]) -> Dict[str, str]:
    if llm is None:
        return {}
    sample_rows = samples[:5]
    prompt = (
        "Infer dataset schema for review mining. Return JSON object with keys "
        "text_col,title_col,rating_col,id_col,domain_col,group_col,split_col,sentiment_col,date_col. "
        f"Columns: {columns}. Sample rows: {sample_rows}"
    )
    data = llm.json_completion(prompt)
    return data if isinstance(data, dict) else {}


def _detect_schema_v2(columns: List[str], samples: List[Dict], llm: Optional[LLMClient] = None) -> Dict:
    mapping = SchemaMapping(
        text_col=_pick(columns, TEXT_KEYS),
        title_col=_pick(columns, TITLE_KEYS),
        rating_col=_pick(columns, RATING_KEYS),
        id_col=_pick(columns, ID_KEYS),
        domain_col=_pick(columns, DOMAIN_KEYS),
        group_col=_pick(columns, GROUP_KEYS),
        split_col=_pick(columns, SPLIT_KEYS),
        sentiment_col=_pick(columns, SENTIMENT_KEYS),
        date_col=_pick(columns, DATE_KEYS),
    )

    if mapping.text_col is None:
        llm_pick = _llm_schema_fallback(columns, samples, llm)
        for field in mapping.__dataclass_fields__.keys():
            val = llm_pick.get(field)
            if isinstance(val, str) and val in columns:
                setattr(mapping, field, val)

    conf = {k: _confidence(getattr(mapping, k)) for k in mapping.__dataclass_fields__.keys()}
    if mapping.text_col:
        conf["text_col"] = 0.95
    return {
        "fingerprint": _fingerprint(columns, samples),
        "mapping": mapping.__dict__,
        "confidence": conf,
    }


def load_file_rows(path: Path) -> tuple[List[Dict[str, Any]], str, List[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f, delimiter=delimiter)]
        cols = list(rows[0].keys()) if rows else []
        return rows, suffix.lstrip("."), cols
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
        cols = sorted({k for r in rows for k in r.keys()})
        return rows, "jsonl", cols
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, list):
            rows = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                rows = [r for r in payload["data"] if isinstance(r, dict)]
            elif isinstance(payload.get("records"), list):
                rows = [r for r in payload["records"] if isinstance(r, dict)]
            else:
                rows = [payload]
        else:
            rows = []
        cols = sorted({k for r in rows for k in r.keys()})
        return rows, "json", cols
    raise ValueError(f"Unsupported file type: {path}")


def _detect_schema_v1(path: Path, columns: List[str], file_type: str) -> DetectedSchema:
    text_col = _pick(columns, TEXT_KEYS)
    aspect_col = _pick(columns, ASPECT_CANDIDATES)
    sentiment_col = _pick(columns, SENTIMENT_KEYS)
    id_col = _pick(columns, ID_KEYS)
    split_col = _pick(columns, SPLIT_KEYS)
    domain_col = _pick(columns, DOMAIN_KEYS)
    evidence_col = _pick(columns, EVIDENCE_CANDIDATES)
    span_from_col = _pick(columns, SPAN_FROM_CANDIDATES)
    span_to_col = _pick(columns, SPAN_TO_CANDIDATES)

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


def detect_schema(*args, **kwargs):
    # New API: detect_schema(columns, samples, llm=...)
    if args and isinstance(args[0], list):
        columns = args[0]
        samples = args[1] if len(args) > 1 else []
        llm = kwargs.get("llm")
        return _detect_schema_v2(columns, samples, llm=llm)

    # Legacy API: detect_schema(path, columns, file_type)
    if len(args) >= 3 and isinstance(args[0], Path):
        path = args[0]
        columns = args[1]
        file_type = args[2]
        return _detect_schema_v1(path, columns, file_type)

    raise TypeError("Unsupported detect_schema call signature")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())
