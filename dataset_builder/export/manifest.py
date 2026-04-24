from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..schemas.artifact_manifest import ArtifactManifest


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
