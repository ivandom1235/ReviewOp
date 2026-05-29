from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from protonet.artifact_guard import _current_dataset_builder_code_hash, assert_verified_artifact
from protonet.config import ECProtoNetV2Config


def test_artifact_guard_fails_on_source_hash_mismatch_unless_bypassed() -> None:
    root = Path("protonet/tests/_tmp_artifact_guard_source_hash")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        (root / "artifact_verification.json").write_text(
            json.dumps(
                {
                    "artifact_status": "pass",
                    "ready_for_protonet": True,
                    "failed_checks": [],
                }
            ),
            encoding="utf-8",
        )
        (root / "source_artifact_consistency.json").write_text(
            json.dumps(
                {
                    "code_hash": "000000000000",
                    "artifact_matches_source": False,
                }
            ),
            encoding="utf-8",
        )

        cfg = ECProtoNetV2Config(require_artifact_pass=True, require_active_contract=False)
        with pytest.raises(Exception):
            assert_verified_artifact(root, cfg)

        report = assert_verified_artifact(root, cfg, allow_stale_source_artifact=True)
        assert report.artifact_status == "pass"
        assert report.ready_for_protonet is True
        assert _current_dataset_builder_code_hash() != "000000000000"
    finally:
        shutil.rmtree(root, ignore_errors=True)
