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


def _label_from_interpretation(
    instance_id: str,
    review_text: str,
    domain: str,
    interp: Dict[str, Any],
    split: str,
    idx: int,
    gold_joint_labels: List[str],
    split_protocol: Dict[str, Any],
    ambiguity_score: float,
    abstain_acceptable: bool,
    novel_aspect_acceptable: bool,
) -> Dict[str, Any]:
    sentiment = _normalize_sentiment(interp.get("sentiment"))
    evidence_text = str(interp.get("evidence_text") or "").strip()
    evidence_span = interp.get("evidence_span")
    if not (isinstance(evidence_span, list) and len(evidence_span) == 2):
        evidence_span = [-1, -1]
    if not evidence_text:
        evidence_text = review_text
    return {
        "example_id": f"{instance_id}_g{idx}",
        "parent_review_id": instance_id,
        "review_text": review_text,
        "evidence_sentence": evidence_text,
        "evidence_text": evidence_text,
        "evidence_span": evidence_span,
        "evidence_fallback_used": evidence_span == [-1, -1],
        "domain": domain,
        "aspect": str(interp.get("aspect_label") or "unknown").strip(),
        "implicit_aspect": str(interp.get("aspect_label") or "unknown").strip(),
        "sentiment": sentiment,
        "label_type": "implicit",
        "confidence": float(interp.get("annotator_support", 1)),
        "split": split,
        "abstain_acceptable": bool(abstain_acceptable),
        "novel_aspect_acceptable": bool(novel_aspect_acceptable),
        "gold_joint_labels": gold_joint_labels,
        "split_protocol": split_protocol,
        "benchmark_ambiguity_score": float(ambiguity_score),
    }


def validate_benchmark_rows(rows: List[Dict[str, Any]], split: str) -> str:
    if not rows:
        raise ValueError(f"No benchmark rows found for split {split}")
    expanded: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        instance_id = str(row.get("instance_id") or "").strip()
        review_text = str(row.get("review_text") or "").strip()
        domain = str(row.get("domain") or "unknown")
        if not instance_id or not review_text:
            raise ValueError(f"benchmark row {index} in split {split} is missing instance_id/review_text")
        interpretations = row.get("gold_interpretations")
        if not isinstance(interpretations, list) or not interpretations:
            raise ValueError(f"benchmark row {index} in split {split} has no gold_interpretations")
        split_protocol = row.get("split_protocol") if isinstance(row.get("split_protocol"), dict) else {}
        ambiguity_score = float(row.get("ambiguity_score", 0.0))
        abstain_acceptable = bool(row.get("abstain_acceptable", False))
        novel_aspect_acceptable = bool(row.get("novel_aspect_acceptable", False))
        gold_joint_labels = []
        for interp in interpretations:
            if not isinstance(interp, dict):
                continue
            aspect = str(interp.get("aspect_label") or "unknown").strip()
            sentiment = _normalize_sentiment(interp.get("sentiment"))
            gold_joint_labels.append(f"{aspect}__{sentiment}")
        for interp_idx, interp in enumerate(interpretations, start=1):
            if not isinstance(interp, dict):
                continue
            expanded.append(
                _label_from_interpretation(
                    instance_id,
                    review_text,
                    domain,
                    interp,
                    split,
                    interp_idx,
                    gold_joint_labels,
                    split_protocol,
                    ambiguity_score,
                    abstain_acceptable,
                    novel_aspect_acceptable,
                )
            )
    rows[:] = expanded
    return "benchmark_examples"


def load_input_dataset(cfg: ProtonetConfig) -> tuple[Dict[str, List[Dict[str, Any]]], DatasetSummary]:
    if cfg.input_type != "benchmark":
        raise ValueError("V6 runtime only supports input_type='benchmark'")
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    detected_format = "unknown"
    for split in VALID_SPLITS:
        path = split_file(cfg.input_dir, split)
        rows = read_split_rows(path, progress_enabled=cfg.progress_enabled)
        detected_format = validate_benchmark_rows(rows, split)
        rows_by_split[split] = rows
    summary = DatasetSummary(
        split_sizes={split: len(rows) for split, rows in rows_by_split.items()},
        detected_format=detected_format,
        input_type=cfg.input_type,
    )
    return rows_by_split, summary
