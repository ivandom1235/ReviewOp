from __future__ import annotations

import sys
import unittest
from pathlib import Path
import importlib.util


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class EpisodeBuilderSelectiveTests(unittest.TestCase):
    def test_episode_row_preserves_evidence_conditioning(self) -> None:
        from episode_builder import _episode_row_from_example

        config_path = Path(__file__).resolve().parents[1] / "code" / "config.py"
        spec = importlib.util.spec_from_file_location("protonet_test_config", config_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ProtonetConfig = module.ProtonetConfig

        cfg = ProtonetConfig()
        row = {
            "example_id": "ex1",
            "parent_review_id": "r1",
            "review_text": "Battery life is poor after update",
            "evidence_sentence": "Battery life is poor",
            "evidence_fallback_used": False,
            "domain": "telecom",
            "domain_family": "telecom",
            "group_id": "g1",
            "aspect": "battery",
            "sentiment": "negative",
            "label_type": "implicit",
            "confidence": 0.91,
            "hardness_tier": "H2",
            "annotation_source": "benchmark_generated",
            "gold_joint_labels": ["battery__negative"],
            "gold_interpretations": [{"aspect_label": "battery", "sentiment": "negative"}],
            "abstain_acceptable": True,
            "ambiguity_type": "semantic",
            "benchmark_ambiguity_score": 0.73,
            "novel_acceptable": True,
            "novel_cluster_id": "novel_abc",
            "novel_alias": "battery drain",
            "novel_evidence_text": "Battery life is poor",
            "split_protocol": {"random": "train", "grouped": "train", "domain_holdout": "val"},
        }

        episode_row = _episode_row_from_example(row, "query", cfg)

        self.assertEqual(episode_row["evidence_sentence"], "Battery life is poor")
        self.assertEqual(episode_row["novel_cluster_id"], "novel_abc")
        self.assertTrue(episode_row["abstain_acceptable"])
        self.assertEqual(episode_row["split_protocol"]["domain_holdout"], "val")


if __name__ == "__main__":
    unittest.main()
