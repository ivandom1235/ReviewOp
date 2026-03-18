"""Validation checks and reporting helpers."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def validate_jsonl(path: Path) -> List[str]:
    errs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                errs.append(f"{path}:{i} invalid json: {exc}")
    return errs


def validate_review_rows(rows: List[Dict]) -> Dict[str, int]:
    counters = Counter()
    seen = set()
    for row in rows:
        if not str(row.get("review_text", "")).strip():
            counters["empty_review_text"] += 1
        rid = str(row.get("id", "")).strip()
        if not rid:
            counters["missing_id"] += 1
        if rid in seen:
            counters["duplicate_id"] += 1
        seen.add(rid)
        labels = row.get("labels", [])
        if not labels:
            counters["no_labels"] += 1
        for label in labels:
            if not str(label.get("aspect", "")).strip():
                counters["invalid_aspect"] += 1
            ev = str(label.get("evidence_sentence", "")).strip()
            if ev and ev not in str(row.get("review_text", "")):
                counters["evidence_not_in_review"] += 1
    return dict(counters)


def aspect_frequency(rows: List[Dict]) -> Dict[str, int]:
    freq = Counter()
    for row in rows:
        for label in row.get("labels", []):
            freq[str(label.get("aspect", "unknown"))] += 1
    return dict(freq)


def few_shot_warnings(episodic_rows: List[Dict], min_examples: int = 5) -> List[str]:
    cnt = Counter(str(r.get("aspect", "unknown")) for r in episodic_rows)
    return [f"aspect '{k}' has only {v} examples" for k, v in sorted(cnt.items()) if v < min_examples]
