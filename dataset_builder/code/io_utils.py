from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


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
        rows = [flatten_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    if "review" in frame.columns:
        return frame
    for alias in _REVIEW_COLUMN_ALIASES:
        if alias in frame.columns:
            # Keep source schema intact for text_column_override compatibility.
            frame = frame.copy()
            frame["review"] = frame[alias]
            return frame
    return frame


def load_inputs(input_dir: Path) -> pd.DataFrame:
    frames = []
    if not input_dir.exists():
        return pd.DataFrame()
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            frame = load_file(path)
            if not frame.empty:
                frame = _normalize_review_columns(frame)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_gold_annotations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".json"}:
        raise ValueError(f"Unsupported gold annotation file type: {path}")
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
            "annotator_id": row.get("annotator_id"),
            "review_status": row.get("review_status", "pending"),
        })
    return cleaned
