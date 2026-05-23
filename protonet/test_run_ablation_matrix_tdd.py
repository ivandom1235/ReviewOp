from __future__ import annotations

import json
import shutil
from pathlib import Path

from protonet.scripts.run_ablation_matrix import _memory_effect_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_memory_effect_report_has_requested_schema_fields() -> None:
    root = Path("protonet/tests/_tmp_run_ablation_matrix")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    no_mem_dir = root / "no_memory" / "domain_holdout"
    promoted_dir = root / "promoted_memory" / "domain_holdout"

    _write_jsonl(
        no_mem_dir / "predictions.jsonl",
        [
            {
                "row_id": "r1",
                "predicted_labels": ["battery_life"],
                "candidates": [{"decision": "accept_known", "unknown_score": 0.10}],
            }
        ],
    )
    _write_jsonl(
        promoted_dir / "predictions.jsonl",
        [
            {
                "row_id": "r1",
                "predicted_labels": ["battery_life", "__open_world__"],
                "candidates": [{"decision": "named_open_world_candidate", "unknown_score": 0.55}],
            }
        ],
    )
    (promoted_dir / "metrics.json").write_text(
        json.dumps({"memory": {"promoted_entries_total": 3, "promoted_matches_used": 1}}),
        encoding="utf-8",
    )

    try:
        report = _memory_effect_report(no_mem_dir, promoted_dir)

        assert report["memory_entry_count"] == 3
        assert report["promoted_matches_used"] == 1
        assert report["changed_prediction_count"] == 1
        assert report["changed_route_count"] == 1
        assert isinstance(report["rows"], list)
    finally:
        shutil.rmtree(root, ignore_errors=True)
