from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def write_sidecar(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
