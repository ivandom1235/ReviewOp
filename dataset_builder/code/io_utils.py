from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def scan_raw_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    supported = {".csv", ".json", ".jsonl"}
    return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported])


def load_rows(path: Path) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        cols = list(rows[0].keys()) if rows else []
        return rows, "csv", cols
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


def ensure_output_dirs(output_dir: Path) -> None:
    for path in [
        output_dir / "reviewlevel" / "normal",
        output_dir / "reviewlevel" / "augmented",
        output_dir / "episodic" / "normal",
        output_dir / "episodic" / "augmented",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def clean_previous_outputs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
