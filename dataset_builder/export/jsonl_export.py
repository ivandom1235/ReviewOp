from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_payload(item) for item in value]
    if isinstance(value, tuple):
        return [to_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: to_payload(item) for key, item in value.items()}
    return value


def write_split_jsonl(output_dir: str | Path, splits: dict[str, list[Any]]) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        rows = splits.get(split, [])
        counts[split] = len(rows)
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_payload(row), ensure_ascii=False, sort_keys=True) + "\n")
    return counts
