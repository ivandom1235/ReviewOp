from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class PipelineRunnerTests(unittest.TestCase):
    def test_run_pipeline_sync_awaits_async_pipeline_result(self) -> None:
        from pipeline_runner import run_pipeline_sync

        async def async_pipeline(cfg):
            await asyncio.sleep(0)
            return {"cfg": cfg, "status": "ok"}

        self.assertEqual(
            run_pipeline_sync("demo-cfg", pipeline=async_pipeline),
            {"cfg": "demo-cfg", "status": "ok"},
        )

    def test_run_pipeline_sync_accepts_sync_pipeline_result(self) -> None:
        from pipeline_runner import run_pipeline_sync

        def sync_pipeline(cfg):
            return {"cfg": cfg, "status": "ok"}

        self.assertEqual(
            run_pipeline_sync("demo-cfg", pipeline=sync_pipeline),
            {"cfg": "demo-cfg", "status": "ok"},
        )

    def test_benchmark_metadata_exposes_semantic_guardrail_metrics(self) -> None:
        from build_dataset import _build_benchmark_instances

        rows = [
            {
                "id": "guardrail-1",
                "domain": "restaurant",
                "review_text": "Service was slow but the food was fresh.",
                "source_text": "Service was slow but the food was fresh.",
                "gold_interpretations": [
                    {
                        "aspect_label": "service_speed",
                        "sentiment": "negative",
                        "evidence_text": "Service was slow",
                        "evidence_span": [-1, -1],
                        "annotator_support": 2,
                        "source": "synthetic",
                        "label_type": "implicit",
                    }
                ],
                "implicit": {
                    "aspects": ["service_speed"],
                    "spans": [
                        {
                            "latent_label": "service_speed",
                            "evidence_text": "Service was slow",
                            "support_type": "exact",
                            "confidence": 0.95,
                        }
                    ],
                },
                "split": "train",
                "group_id": "restaurant_guardrail_1",
                "abstain_acceptable": True,
                "novel_acceptable": False,
            }
        ]
        assignments = {"guardrail-1": {"random": "train", "grouped": "train", "domain_holdout": "train"}}
        _, metadata, _ = _build_benchmark_instances(rows, assignments, enforce_registry_membership=False)

        self.assertIn("implicit_purity_rate", metadata)
        self.assertIn("ontology_compatibility_rate", metadata)
        self.assertIn("light_repair_count", metadata)
        self.assertIn("hard_span_failure_count", metadata)


if __name__ == "__main__":
    unittest.main()
