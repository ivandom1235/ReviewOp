from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .config import ProtonetConfig, split_file
    from .progress import track
except ImportError:
    from config import ProtonetConfig, split_file
    from progress import track


VALID_SPLITS = ("train", "val", "test")


@dataclass
class DatasetSummary:
    split_sizes: Dict[str, int]
    detected_format: str
    input_type: str


def _normalize_sentiment(value: Any, fallback: str = "neutral") -> str:
    candidate = value
    if isinstance(candidate, list):
        candidate = candidate[0] if candidate else fallback
    sentiment = str(candidate or fallback).strip().lower()
    return sentiment or fallback


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object rows in {path}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_split_rows(path: Path, *, progress_enabled: bool) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input split file: {path}")
    rows = load_jsonl(path)
    return list(track(rows, total=len(rows), desc=f"load:{path.stem}", enabled=progress_enabled))


def validate_reviewlevel_rows(rows: List[Dict[str, Any]], split: str) -> None:
    for index, row in enumerate(rows):
        labels = row.get("labels")
        if (labels is None or not isinstance(labels, list) or not labels) and isinstance(row.get("implicit"), dict):
            implicit = row.get("implicit", {})
            labels = [
                {
                    "aspect": aspect,
                    "implicit_aspect": aspect,
                    "sentiment": _normalize_sentiment(
                        implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))
                    ),
                    "confidence": float(implicit.get("aspect_confidence", {}).get(aspect, implicit.get("avg_confidence", 0.5))),
                    "evidence_sentence": row.get("source_text", row.get("review_text", "")),
                }
                for aspect in implicit.get("aspects", []) or []
            ]
            row["labels"] = labels
        if "labels" not in row or not isinstance(row["labels"], list) or not row["labels"]:
            raise ValueError(f"reviewlevel row {index} in split {split} has no labels")
        for label in row["labels"]:
            if isinstance(label, dict):
                label["sentiment"] = _normalize_sentiment(label.get("sentiment"))
        if not row.get("review_text") and not row.get("clean_text") and not row.get("source_text"):
            raise ValueError(f"reviewlevel row {index} in split {split} is missing review text")
        if row.get("split") and str(row["split"]).lower() != split:
            raise ValueError(f"reviewlevel row {index} in split {split} declares split {row['split']}")


def validate_episodic_rows(rows: List[Dict[str, Any]], split: str) -> str:
    if not rows:
        raise ValueError(f"No episodic rows found for split {split}")
    first = rows[0]
    if "implicit" in first and isinstance(first.get("implicit"), dict):
        expanded: List[Dict[str, Any]] = []
        for row in rows:
            implicit = row.get("implicit", {})
            for idx, aspect in enumerate(implicit.get("aspects", []) or [], start=1):
                expanded.append(
                    {
                        "example_id": f"{row.get('id')}_e{idx}",
                        "parent_review_id": row.get("id"),
                        "review_text": row.get("source_text", row.get("review_text", "")),
                        "evidence_sentence": row.get("source_text", row.get("review_text", "")),
                        "domain": row.get("domain", "unknown"),
                        "aspect": aspect,
                        "implicit_aspect": aspect,
                        "sentiment": _normalize_sentiment(
                            implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))
                        ),
                        "label_type": "implicit",
                        "confidence": float(implicit.get("aspect_confidence", {}).get(aspect, implicit.get("avg_confidence", 0.5))),
                        "split": row.get("split", split),
                    }
                )
        rows[:] = expanded
        return "examples"
    if "support_set" in first and "query_set" in first:
        for index, row in enumerate(rows):
            if not isinstance(row.get("support_set"), list) or not isinstance(row.get("query_set"), list):
                raise ValueError(f"episode row {index} in split {split} has invalid support/query sets")
        return "episodes"
    required = {"example_id", "parent_review_id", "review_text", "aspect", "sentiment"}
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"episodic row {index} in split {split} is missing fields: {missing}")
        if row.get("split") and str(row["split"]).lower() != split:
            raise ValueError(f"episodic row {index} in split {split} declares split {row['split']}")
    return "examples"


def load_input_dataset(cfg: ProtonetConfig) -> tuple[Dict[str, List[Dict[str, Any]]], DatasetSummary]:
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    detected_format = "unknown"
    for split in VALID_SPLITS:
        path = split_file(cfg.input_dir, split)
        rows = read_split_rows(path, progress_enabled=cfg.progress_enabled)
        if cfg.input_type == "reviewlevel":
            validate_reviewlevel_rows(rows, split)
            detected_format = "reviewlevel"
        else:
            detected_format = validate_episodic_rows(rows, split)
        rows_by_split[split] = rows
    summary = DatasetSummary(
        split_sizes={split: len(rows) for split, rows in rows_by_split.items()},
        detected_format=detected_format,
        input_type=cfg.input_type,
    )
    return rows_by_split, summary
