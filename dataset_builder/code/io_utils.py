from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


SUPPORTED_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".xml"}


def flatten_dict(payload: dict) -> dict:
    flat: dict = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}_{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat


def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else None, engine="python")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.DataFrame([flatten_dict(row) if isinstance(row, dict) else {"value": row} for row in payload])
        else:
            frame = pd.DataFrame([flatten_dict(payload if isinstance(payload, dict) else {"value": payload})])
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [flatten_dict(json.loads(line)) for line in f if line.strip()]
        frame = pd.DataFrame(rows)
    elif suffix == ".xml":
        root = ET.parse(path).getroot()
        rows = []
        for child in list(root):
            row = {f"attr_{k}": v for k, v in child.attrib.items()}
            for sub in list(child):
                row[sub.tag] = (sub.text or "").strip()
            if not row and (child.text or "").strip():
                row[child.tag] = (child.text or "").strip()
            rows.append(row)
        frame = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported input file: {path}")
    if not frame.empty:
        frame = frame.copy()
        frame["source_file"] = path.name
    return frame


_REVIEW_COLUMN_ALIASES = [
    "ReviewText", "review_text", "text", "Text", "text_",
    "comment", "Comment", "body", "Body",
    "content", "Content", "review_body", "ReviewBody",
]


def _normalize_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # We copy the frame to ensure a separate object for every process.
    frame = frame.copy()
    for alias in _REVIEW_COLUMN_ALIASES:
        if alias in frame.columns and "text" not in frame.columns:
            frame["text"] = frame[alias]
            break
    # Default column name for the primary review text
    if "text" not in frame.columns and not frame.empty:
        # Fallback to the first object-type column if no alias matches
        object_cols = frame.select_dtypes(include=["object", "string"]).columns
        if len(object_cols) > 0:
            frame["text"] = frame[object_cols[0]]
            
    # Ensure source_text is also populated as it's used in some downstream logic
    if "text" in frame.columns:
        frame["source_text"] = frame["text"]
    return frame


def _load_and_normalize(path: Path) -> pd.DataFrame:
    # This helper must be at module level for ProcessPoolExecutor to pickle it.
    try:
        frame = load_file(path)
        if not frame.empty:
            return _normalize_review_columns(frame)
        return frame
    except Exception as e:
        print(f"[!] Error loading {path.name}: {e}")
        return pd.DataFrame()


def load_inputs(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists():
        return pd.DataFrame()
        
    files = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    
    if not files:
        return pd.DataFrame()

    # Use multiprocessing only for multiple files to avoid overhead
    if len(files) > 2:
        # We use a reasonable worker count to avoid over-saturating the user's "slow" system
        num_workers = min(len(files), multiprocessing.cpu_count())
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                frames = list(executor.map(_load_and_normalize, files))
        except Exception as e:
            print(f"[!] Parallel input loading failed, falling back to serial mode: {e}")
            frames = [_load_and_normalize(f) for f in files]
    else:
        frames = [_load_and_normalize(f) for f in files]
        
    valid_frames = [f for f in frames if not f.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True, sort=False)


def load_gold_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".json"}:
        raise ValueError(f"Unsupported gold annotation file type: {path}")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
    cleaned: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels = row.get("gold_labels")
        cleaned.append({
            "record_id": row.get("record_id"),
            "domain": row.get("domain"),
            "text": row.get("text"),
            "gold_labels": labels if isinstance(labels, list) else [],
            "gold_interpretations": row.get("gold_interpretations") if isinstance(row.get("gold_interpretations"), list) else [],
            "abstain_acceptable": bool(row.get("abstain_acceptable", False)),
            "novel_aspect_acceptable": bool(row.get("novel_aspect_acceptable", False)),
            "annotator_id": row.get("annotator_id"),
            "annotator_support": row.get("annotator_support"),
            "review_status": row.get("review_status", "pending"),
        })
    return cleaned
