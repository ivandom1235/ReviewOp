from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_dataset import _benchmark_export_priority
from analyze_reports import _build_scorecard


def _make_benchmark_row(
    row_id: str,
    *,
    implicit_count: int,
    explicit_count: int,
    hardness_tier: str = "H2",
    abstain_acceptable: bool = False,
    novel_acceptable: bool = False,
) -> dict[str, object]:
    gold_interpretations = []
    implicit_grounded = []
    explicit_grounded = []

    for idx in range(implicit_count):
        item = {
            "aspect_label": f"implicit-{idx}",
            "sentiment": "positive",
            "evidence_text": f"implicit evidence {idx}",
            "source": "rule",
            "evidence_mode": "implicit",
        }
        gold_interpretations.append(item)
        implicit_grounded.append(item)

    for idx in range(explicit_count):
        item = {
            "aspect_label": f"explicit-{idx}",
            "sentiment": "positive",
            "evidence_text": f"explicit evidence {idx}",
            "source": "manual",
            "evidence_mode": "explicit",
        }
        gold_interpretations.append(item)
        explicit_grounded.append(item)

    return {
        "instance_id": row_id,
        "record_id": row_id,
        "review_text": "same text",
        "domain": "laptop",
        "domain_family": "electronics",
        "group_id": "group-1",
        "gold_interpretations": gold_interpretations,
        "implicit_grounded_interpretations": implicit_grounded,
        "explicit_grounded_interpretations": explicit_grounded,
        "abstain_acceptable": abstain_acceptable,
        "novel_acceptable": novel_acceptable,
        "hardness_tier": hardness_tier,
        "split_protocol": {"random": "train", "grouped": "train", "domain_holdout": "train"},
    }


class BenchmarkCleanupTests(unittest.TestCase):
    def test_benchmark_priority_prefers_implicit_rich_duplicates(self) -> None:
        explicit_heavy = _make_benchmark_row("row-explicit", implicit_count=1, explicit_count=4)
        implicit_rich = _make_benchmark_row("row-implicit", implicit_count=4, explicit_count=1)

        ordered = sorted([explicit_heavy, implicit_rich], key=_benchmark_export_priority, reverse=True)

        self.assertEqual(ordered[0]["instance_id"], "row-implicit")

    def test_scorecard_reports_viability_guard_when_train_is_saved_from_collapse(self) -> None:
        build = {
            "row_counts": {"train_export": 8},
            "train_target_stats": {"target_min_rows": 280, "target_max_rows": 525, "size_within_target_range": False},
            "train_domain_leakage_row_rate": 0.0,
            "train_general_dominance_rate": 0.0,
            "train_viability_guard_triggered": True,
            "output_version": "v6",
            "pipeline_version": "6.0-v6-only",
            "run_profile": "research",
            "config": {"train_max_positive_ratio": 0.5},
            "benchmark_gold_eval": {"has_gold_interpretations": True, "implicit_purity_rate": 0.5, "ontology_compatibility_rate": 1.0},
            "train_topup_stats": {},
            "train_sentiment_constraints": {},
            "grounded_prediction_rate": 0.98,
            "strict_quality": {},
            "output_quality": {},
        }

        scorecard = _build_scorecard(build, {}, None)

        self.assertIn(
            "Train-size viability guard prevented the sentiment balancer from collapsing the export further.",
            scorecard["root_cause_attribution"],
        )


if __name__ == "__main__":
    unittest.main()
