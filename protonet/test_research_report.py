from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROTONET_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROTONET_ROOT / "code"
for path in (PROTONET_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_report import build_research_pack, write_research_pack


class ResearchReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = PROTONET_ROOT / "output" / "_tmp_research_report_test"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmp_dir, ignore_errors=True))

    def test_builds_ablation_and_counterfactual_pack(self) -> None:
        report_dir = self.tmp_dir / "metadata"
        preds_dir = self.tmp_dir / "output" / "predictions"
        report_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps({"config": {"seed": 7}, "input_summary": {"extra_artifacts": {"counterfactual_pairs": {"total": 2}}}}),
            encoding="utf-8",
        )
        rows = [
            {
                "pred_label": "battery__positive",
                "correct": True,
                "confidence": 0.9,
                "abstained": False,
                "post_aspect_selected_aspects": ["battery"],
                "benchmark_ambiguity_score": 0.1,
                "novelty_score": 0.2,
                "counterfactual_group_id": "cf_1",
                "counterfactual_role": "original",
            },
            {
                "pred_label": "screen__negative",
                "correct": False,
                "confidence": 0.18,
                "abstained": True,
                "post_aspect_selected_aspects": [],
                "benchmark_ambiguity_score": 0.8,
                "novelty_score": 0.9,
                "counterfactual_group_id": "cf_1",
                "counterfactual_role": "counterfactual",
            },
        ]
        (preds_dir / "test_predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        pack = build_research_pack(report_dir=report_dir)
        self.assertIn("ablations", pack)
        self.assertIn("test", pack["splits"])
        self.assertEqual(pack["counterfactual"]["from_report"]["total"], 2)
        self.assertGreaterEqual(pack["counterfactual"]["observed"]["evaluated_groups"], 1)

        output = self.tmp_dir / "research_pack.json"
        write_research_pack(pack, output)
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
