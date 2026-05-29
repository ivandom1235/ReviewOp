from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_manifest_checksums_match_current_files() -> None:
    root = Path("dataset_builder/output")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    checksums = manifest.get("artifact_checksums", {}) if isinstance(manifest, dict) else {}
    if not isinstance(checksums, dict):
        return

    for filename, expected in checksums.items():
        path = root / filename
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_hash = str(expected).split(":")[-1]
        assert expected_hash == actual, f"Checksum mismatch for {filename}"
