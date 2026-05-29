from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json"}
EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "output",
    "outputs",
    "results",
    "protonet_results",
    "metadata",
    "runs",
}


def sha256_source_tree(path: str | Path) -> str:
    base = Path(path)
    h = hashlib.sha256()
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if set(rel.parts) & EXCLUDE_DIRS:
            continue
        if p.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        rel_norm = str(rel).replace("\\", "/")
        h.update(rel_norm.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()
