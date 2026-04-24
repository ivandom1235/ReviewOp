from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from protonet.code.config import ProtonetConfig
from protonet.code.dataset_reader import validate_benchmark_rows
from protonet.code.evaluator import evaluate_episodes


class DatasetReaderSchemaAlignmentTests(unittest.TestCase):
    def test_maps_builder_novelty_status_and_carries_metadata(self) -> None:
        rows, _ = validate_benchmark_rows(
            [
                {
                    "review_id": "r1",
                    "group_id": "g1",
                    "domain": "electronics",
                    "review_text": "The hinge sparks when opened.",
                    "novelty_status": "novel",
                    "abstain_acceptable": True,
                    "hardness_tier": "H3",
                    "gold_interpretations": [
                        {
                            "aspect_raw": "hinge sparks",
                            "aspect_canonical": "hinge_sparks",
                            "sentiment": "negative",
                            "label_type": "implicit",
                            "source_type": "implicit_learned",
                            "evidence_text": "hinge sparks",
                            "evidence_span": [4, 16],
                        }
                    ],
                }
            ],
            "test",
        )

        self.assertEqual(rows[0]["novelty_status"], "novel")
        self.assertTrue(rows[0]["novel_acceptable"])
        self.assertTrue(rows[0]["abstain_acceptable"])
        self.assertEqual(rows[0]["hardness_tier"], "H3")
        self.assertEqual(rows[0]["source_type"], "implicit_learned")


class FakeModel:
    def __init__(self) -> None:
        self.temperature = torch.tensor(1.0)

    def eval(self) -> None:
        return None

    def episode_forward(self, episode):
        return SimpleNamespace(
            query_embeddings=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            prototypes=torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32),
            probabilities=torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            targets=torch.tensor([0], dtype=torch.long),
            ordered_labels=["battery__negative", "display__negative"],
            predictions=["battery__negative"],
        )


class EvaluatorSchemaAlignmentTests(unittest.TestCase):
    def test_reports_skip_reasons_for_missing_novelty_and_protocol_labels(self) -> None:
        cfg = ProtonetConfig(no_progress=True)
        metrics, _ = evaluate_episodes(
            FakeModel(),
            [
                {
                    "episode_id": "e1",
                    "query_set": [
                        {
                            "review_text": "Battery is weak.",
                            "evidence_text": "Battery",
                            "gold_joint_labels": ["battery__negative"],
                            "novelty_status": "known",
                            "source_type": "explicit",
                            "hardness_tier": "H0",
                            "abstain_acceptable": False,
                        }
                    ],
                }
            ],
            cfg,
            "test",
            compute_curves=False,
        )

        self.assertTrue(metrics["known_vs_novel_not_applicable"])
        self.assertEqual(metrics["novelty_evaluation_skipped_reason"], "no_novel_positive_examples")
        self.assertEqual(metrics["protocol_breakdown"]["domain_holdout"]["status"], "skipped")
        self.assertEqual(metrics["protocol_breakdown"]["domain_holdout"]["reason"], "missing split_protocol")
        self.assertEqual(metrics["source_type_breakdown"]["explicit"]["count"], 1)
        self.assertEqual(metrics["hardness_breakdown"]["H0"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
