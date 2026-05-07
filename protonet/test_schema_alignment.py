from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from protonet.code.config import ProtonetConfig
from protonet.code.dataset_reader import load_input_dataset, validate_benchmark_rows
from protonet.code.evaluator import evaluate_episodes
from protonet.code.selective_decisions import combine_routing_score, decide_prediction_state


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

    def test_prediction_state_helper_matches_selective_routing_shape(self) -> None:
        state = decide_prediction_state(
            novelty_score=0.92,
            selective_confidence=0.83,
            abstain_threshold=0.1,
            known_threshold=0.5,
            novel_threshold=0.8,
        )

        self.assertEqual(state["decision"], "novel")
        self.assertTrue(state["route_novel"])
        self.assertIn("selective_score", state)

    def test_combined_routing_score_penalizes_contradiction_signal(self) -> None:
        base = combine_routing_score(
            prototype_similarity=0.9,
            evidence_support=0.8,
            verifier_support=0.7,
            memory_support=0.6,
            ambiguity_penalty=0.1,
            novelty_risk=0.1,
        )
        penalized = combine_routing_score(
            prototype_similarity=0.9,
            evidence_support=0.8,
            verifier_support=0.7,
            memory_support=0.6,
            ambiguity_penalty=0.1,
            novelty_risk=0.1,
            contradiction_score=1.0,
        )

        self.assertLess(penalized, base)


class DomainHoldoutLoaderTests(unittest.TestCase):
    def test_load_input_dataset_picks_up_optional_domain_holdout_and_counterfactuals(self) -> None:
        root = Path("protonet/output/_tmp_loader_test")
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            root.rmdir()
        root.mkdir(parents=True, exist_ok=True)
        try:
            for split, text, domain in (
                ("train", "Battery is weak.", "electronics"),
                ("val", "Screen is bright.", "electronics"),
                ("test", "Service was slow.", "restaurant"),
            ):
                (root / f"{split}.jsonl").write_text(
                    json.dumps(
                        {
                            "review_id": f"{split}-1",
                            "group_id": f"g-{split}",
                            "domain": domain,
                            "domain_family": domain,
                            "review_text": text,
                            "gold_interpretations": [
                                {
                                    "aspect_raw": "battery",
                                    "aspect_canonical": "battery_life",
                                    "latent_family": "battery",
                                    "label_type": "explicit",
                                    "sentiment": "negative",
                                    "evidence_text": text.split(".")[0],
                                    "evidence_span": [0, len(text.split(".")[0])],
                                    "source_type": "explicit",
                                    "support_type": "exact",
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            domain_holdout_dir = root / "domain_holdout"
            domain_holdout_dir.mkdir()
            (domain_holdout_dir / "domain_holdout.jsonl").write_text(
                json.dumps(
                    {
                        "review_id": "dh-1",
                        "group_id": "g-dh",
                        "domain": "electronics",
                        "domain_family": "electronics",
                        "review_text": "Battery lasted longer than expected.",
                        "gold_interpretations": [
                            {
                                "aspect_raw": "battery",
                                "aspect_canonical": "battery_life",
                                "latent_family": "battery",
                                "label_type": "explicit",
                                "sentiment": "positive",
                                "evidence_text": "Battery",
                                "evidence_span": [0, 7],
                                "source_type": "explicit",
                                "support_type": "exact",
                            }
                        ],
                        "split_protocol": {"random": "unused", "grouped": "unused", "domain_holdout": "test"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "counterfactual_pairs.json").write_text(
                json.dumps([{"counterfactual_group_id": "cf_1", "source_text": "Battery is weak.", "counterfactual_text": "Screen is weak."}]),
                encoding="utf-8",
            )

            cfg = ProtonetConfig(input_dir=root, no_progress=True)
            rows_by_split, summary = load_input_dataset(cfg)

            self.assertIn("domain_holdout", rows_by_split)
            self.assertEqual(summary.split_sizes["domain_holdout"], 1)
            self.assertEqual(summary.extra_artifacts["counterfactual_pairs"]["count"], 1)
            self.assertEqual(summary.extra_artifacts["domain_holdout"]["split_sizes"]["domain_holdout"], 1)
        finally:
            if root.exists():
                for child in sorted(root.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                root.rmdir()

    def test_load_input_dataset_reads_split_based_domain_holdout_layout(self) -> None:
        root = Path("protonet/output/_tmp_loader_split_layout")
        if root.exists():
            for child in sorted(root.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            root.rmdir()
        root.mkdir(parents=True, exist_ok=True)
        try:
            for split in ("train", "val", "test"):
                (root / f"{split}.jsonl").write_text(
                    json.dumps(
                        {
                            "review_id": f"{split}-1",
                            "group_id": f"g-{split}",
                            "domain": "electronics",
                            "domain_family": "electronics",
                            "review_text": f"{split} row.",
                            "gold_interpretations": [
                                {
                                    "aspect_raw": "battery",
                                    "aspect_canonical": "battery_life",
                                    "latent_family": "battery",
                                    "label_type": "explicit",
                                    "sentiment": "negative",
                                    "evidence_text": "battery",
                                    "evidence_span": [0, 7],
                                    "source_type": "explicit",
                                    "support_type": "exact",
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            domain_holdout_dir = root / "domain_holdout"
            domain_holdout_dir.mkdir()
            for split in ("train", "val", "test"):
                (domain_holdout_dir / f"{split}.jsonl").write_text(
                    json.dumps(
                        {
                            "review_id": f"dh-{split}-1",
                            "group_id": f"dh-{split}",
                            "domain": "restaurant" if split == "test" else "electronics",
                            "domain_family": "electronics",
                            "review_text": f"domain holdout {split}.",
                            "gold_interpretations": [
                                {
                                    "aspect_raw": "service",
                                    "aspect_canonical": "service_speed",
                                    "latent_family": "service",
                                    "label_type": "explicit",
                                    "sentiment": "negative",
                                    "evidence_text": "service",
                                    "evidence_span": [0, 7],
                                    "source_type": "explicit",
                                    "support_type": "exact",
                                }
                            ],
                            "split_protocol": {"random": "unused", "grouped": "unused", "domain_holdout": split},
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            (root / "counterfactual_pairs.json").write_text(
                json.dumps([{"counterfactual_group_id": "cf_1", "source_text": "Battery is weak.", "counterfactual_text": "Screen is weak."}]),
                encoding="utf-8",
            )

            cfg = ProtonetConfig(input_dir=root, no_progress=True)
            rows_by_split, summary = load_input_dataset(cfg)

            self.assertIn("domain_holdout", rows_by_split)
            self.assertEqual(summary.split_sizes["domain_holdout"], 3)
            self.assertEqual(summary.extra_artifacts["counterfactual_pairs"]["count"], 1)
            self.assertEqual(summary.extra_artifacts["domain_holdout"]["split_sizes"]["train"], 1)
            self.assertEqual(summary.extra_artifacts["domain_holdout"]["split_sizes"]["val"], 1)
            self.assertEqual(summary.extra_artifacts["domain_holdout"]["split_sizes"]["test"], 1)
        finally:
            if root.exists():
                for child in sorted(root.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                root.rmdir()


if __name__ == "__main__":
    unittest.main()
