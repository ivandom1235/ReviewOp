from __future__ import annotations

from pathlib import Path
from typing import Any

from utils import write_jsonl


def write_split_outputs(base_dir: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    for split, rows in payload.items():
        write_jsonl(base_dir / f"{split}.jsonl", rows)


def write_compat_exports(target_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        write_jsonl(target_dir / f"{split}.jsonl", rows)


def write_named_outputs(base_dir: Path, payload: dict[str, list[dict[str, Any]]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in payload.items():
        write_jsonl(base_dir / f"{name}.jsonl", rows)
