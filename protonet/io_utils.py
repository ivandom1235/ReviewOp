from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: str | Path, suffixes: tuple[str, ...] = (".json", ".jsonl", ".yaml", ".yml")) -> str:
    path = Path(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            h.update(str(p.relative_to(path)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def source_code_hash(path: str | Path) -> str:
    base = Path(path)
    include = {".py", ".json", ".yaml", ".yml", ".toml"}
    skip_parts = {"output", "outputs", "runs", "__pycache__", ".pytest_cache"}
    h = hashlib.sha256()
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip_parts for part in p.parts):
            continue
        if p.suffix.lower() not in include:
            continue
        rel = p.relative_to(base).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()
