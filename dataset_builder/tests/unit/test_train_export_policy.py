from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from build_dataset import (
    _apply_train_fallback_general_policy,
    _apply_train_review_filter,
    _apply_train_sentiment_balance,
    _strict_quality_metrics,
    _strict_row_passes,
    _strict_topup_recovery,
    _strict_train_domain_leakage_filter,
    _train_domain_leakage_metrics,
)


def _row(
    row_id: str,
    *,
    domain: str,
    language: str,
    aspects: list[str],
    sentiment: str,
    needs_review: bool | None = None,
    review_reason: str | None = None,
    support_type: str = "exact",
    confidence: float = 0.9,
) -> dict:
    if needs_review is None:
        needs_review = aspects == ["general"]
    aspect_conf = {aspect: confidence for aspect in aspects if aspect != "general"}
    spans = []
    if aspects != ["general"]:
        spans = [{"support_type": support_type, "confidence": confidence}]
    return {
        "id": row_id,
        "domain": domain,
        "language": language,
        "source_text": f"{domain} sample {row_id}",
        "implicit": {
            "aspects": aspects,
            "dominant_sentiment": sentiment,
            "needs_review": needs_review,
            "review_reason": review_reason,
            "aspect_confidence": aspect_conf,
            "spans": spans,
            "hardness_tier": "H2" if support_type != "exact" else "H1",
            "implicit_quality_tier": "strict_pass" if not needs_review else "needs_review",
        },
    }


class TrainExportPolicyTests(unittest.TestCase):
    def test_cap_policy_is_deterministic_and_group_bounded(self) -> None:
        rows = [
            _row("a1", domain="restaurant", language="en", aspects=["general"], sentiment="neutral"),
            _row("a2", domain="restaurant", language="en", aspects=["general"], sentiment="neutral"),
            _row("a3", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
            _row("a4", domain="restaurant", language="en", aspects=["service quality"], sentiment="negative"),
            _row("b1", domain="laptop", language="en", aspects=["general"], sentiment="neutral"),
            _row("b2", domain="laptop", language="en", aspects=["general"], sentiment="neutral"),
            _row("b3", domain="laptop", language="en", aspects=["performance"], sentiment="positive"),
            _row("b4", domain="laptop", language="en", aspects=["value"], sentiment="negative"),
        ]
        kept1, stats1 = _apply_train_fallback_general_policy(rows, policy="cap", cap_ratio=0.25, seed=42)
        kept2, stats2 = _apply_train_fallback_general_policy(rows, policy="cap", cap_ratio=0.25, seed=42)
        self.assertEqual([r["id"] for r in kept1], [r["id"] for r in kept2])
        self.assertEqual(stats1, stats2)
        kept_general = [r for r in kept1 if r["implicit"]["aspects"] == ["general"]]
        self.assertEqual(len(kept_general), 2)
        self.assertEqual(stats1["train_general_rows_before_policy"], 4)
        self.assertEqual(stats1["train_general_rows_after_policy"], 2)

    def test_drop_policy_removes_only_fallback_general(self) -> None:
        rows = [
            _row("a1", domain="restaurant", language="en", aspects=["general"], sentiment="neutral"),
            _row("a2", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
        ]
        kept, stats = _apply_train_fallback_general_policy(rows, policy="drop", cap_ratio=0.15, seed=42)
        self.assertEqual([r["id"] for r in kept], ["a2"])
        self.assertEqual(stats["train_general_rows_before_policy"], 1)
        self.assertEqual(stats["train_general_rows_after_policy"], 0)

    def test_cap_neutral_balance_applies_on_train_rows(self) -> None:
        rows = [
            _row("a1", domain="restaurant", language="en", aspects=["food quality"], sentiment="neutral"),
            _row("a2", domain="restaurant", language="en", aspects=["service quality"], sentiment="neutral"),
            _row("a3", domain="restaurant", language="en", aspects=["service quality"], sentiment="positive"),
            _row("a4", domain="restaurant", language="en", aspects=["value"], sentiment="negative"),
        ]
        balanced, before_counts, after_counts, _ = _apply_train_sentiment_balance(
            rows,
            mode="cap_neutral",
            neutral_cap_ratio=0.25,
            min_negative_ratio=0.12,
            min_positive_ratio=0.12,
            max_positive_ratio=0.5,
            neutral_max_ratio=0.58,
            seed=42,
        )
        self.assertEqual(before_counts["neutral"], 2)
        self.assertEqual(after_counts.get("neutral", 0), 1)
        self.assertEqual(len(balanced), 3)

    def test_review_filter_drops_needs_review_rows(self) -> None:
        rows = [
            _row("a1", domain="restaurant", language="en", aspects=["general"], sentiment="neutral"),
            _row("a2", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
            _row("a3", domain="restaurant", language="en", aspects=["service quality"], sentiment="negative"),
        ]
        kept, stats = _apply_train_review_filter(rows, mode="drop_needs_review")
        self.assertEqual([row["id"] for row in kept], ["a2", "a3"])
        self.assertEqual(stats["train_review_rows_before_filter"], 3)
        self.assertEqual(stats["train_review_rows_after_filter"], 2)

    def test_negative_floor_caps_positive_when_negative_is_underrepresented(self) -> None:
        rows = [
            _row("n1", domain="laptop", language="en", aspects=["performance"], sentiment="negative"),
            _row("p1", domain="laptop", language="en", aspects=["performance"], sentiment="positive"),
            _row("p2", domain="laptop", language="en", aspects=["value"], sentiment="positive"),
            _row("p3", domain="laptop", language="en", aspects=["power"], sentiment="positive"),
            _row("p4", domain="laptop", language="en", aspects=["display quality"], sentiment="positive"),
            _row("u1", domain="laptop", language="en", aspects=["value"], sentiment="neutral"),
        ]
        balanced, _, after_counts, _ = _apply_train_sentiment_balance(
            rows,
            mode="cap_neutral_with_dual_floor",
            neutral_cap_ratio=1.0,
            min_negative_ratio=0.25,
            min_positive_ratio=0.25,
            max_positive_ratio=0.5,
            neutral_max_ratio=0.58,
            seed=42,
        )
        negative = after_counts.get("negative", 0)
        total = len(balanced)
        self.assertGreaterEqual(negative / total, 0.25)

    def test_reasoned_strict_review_filter_keeps_only_qualified_rows(self) -> None:
        rows = [
            _row("a1", domain="restaurant", language="en", aspects=["general"], sentiment="neutral", needs_review=True, review_reason="fallback_general"),
            _row("a2", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive", needs_review=True, review_reason="weak_support", support_type="near_exact", confidence=0.7),
            _row("a3", domain="restaurant", language="en", aspects=["performance"], sentiment="positive", needs_review=True, review_reason="weak_support", support_type="near_exact", confidence=0.9),
            _row("a4", domain="restaurant", language="en", aspects=["service quality"], sentiment="positive", needs_review=False, review_reason=None),
        ]
        kept, stats = _apply_train_review_filter(
            rows,
            mode="reasoned_strict",
            candidate_aspects_by_domain={"restaurant": ["food", "service"]},
            min_confidence=0.58,
            accepted_support_types=("exact", "near_exact", "gold"),
        )
        self.assertEqual({row["id"] for row in kept}, {"a2", "a4"})
        self.assertEqual(stats["train_review_rows_before_filter"], 4)
        self.assertEqual(stats["train_review_rows_after_filter"], 2)

    def test_bounded_balance_caps_positive_and_neutral(self) -> None:
        rows = [
            _row("n1", domain="laptop", language="en", aspects=["performance"], sentiment="negative"),
            _row("p1", domain="laptop", language="en", aspects=["performance"], sentiment="positive"),
            _row("p2", domain="laptop", language="en", aspects=["value"], sentiment="positive"),
            _row("p3", domain="laptop", language="en", aspects=["power"], sentiment="positive"),
            _row("p4", domain="laptop", language="en", aspects=["display quality"], sentiment="positive"),
            _row("u1", domain="laptop", language="en", aspects=["value"], sentiment="neutral"),
            _row("u2", domain="laptop", language="en", aspects=["display quality"], sentiment="neutral"),
            _row("u3", domain="laptop", language="en", aspects=["performance"], sentiment="neutral"),
        ]
        balanced, _, _, constraints = _apply_train_sentiment_balance(
            rows,
            mode="cap_neutral_with_dual_floor",
            neutral_cap_ratio=1.0,
            min_negative_ratio=0.12,
            min_positive_ratio=0.12,
            max_positive_ratio=0.5,
            neutral_max_ratio=0.58,
            seed=42,
        )
        self.assertLessEqual(constraints["achieved"]["positive_ratio"], 0.5)
        self.assertLessEqual(constraints["achieved"]["neutral_ratio"], 0.58)
        self.assertGreater(len(balanced), 0)

    def test_strict_topup_recovery_respects_constraints(self) -> None:
        current_rows = [
            _row("r1", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
            _row("r2", domain="restaurant", language="en", aspects=["service quality"], sentiment="negative"),
        ]
        candidates = [
            _row("r3", domain="restaurant", language="en", aspects=["food quality"], sentiment="neutral", needs_review=True, review_reason="weak_support", support_type="near_exact", confidence=0.9),
            _row("r4", domain="restaurant", language="en", aspects=["general"], sentiment="neutral", needs_review=True, review_reason="fallback_general", support_type="exact", confidence=0.9),
            _row("r5", domain="restaurant", language="en", aspects=["performance"], sentiment="positive", needs_review=True, review_reason="weak_support", support_type="near_exact", confidence=0.9),
        ]
        recovered_rows, stats = _strict_topup_recovery(
            train_rows=current_rows,
            candidate_rows=candidates,
            mode="strict_topup",
            target_min_rows=4,
            confidence_threshold=0.58,
            stage_b_confidence_threshold=0.54,
            stage_c_confidence_threshold=0.52,
            staged_recovery=True,
            allow_weak_support_in_stage_c=True,
            accepted_support_types=("exact", "near_exact", "gold"),
            candidate_aspects_by_domain={"restaurant": ["food", "service", "value"]},
            seed=42,
        )
        self.assertEqual(stats["topup_rows_added"], 1)
        self.assertIn("r3", {row["id"] for row in recovered_rows})
        self.assertNotIn("r5", {row["id"] for row in recovered_rows})
        self.assertIn("train_topup_rejection_breakdown", stats)
        self.assertIn("topup_effectiveness", stats)

    def test_strict_topup_stages_progress_when_shortfall_remains(self) -> None:
        current_rows = [
            _row("r1", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
        ]
        candidates = [
            _row(
                "r2",
                domain="restaurant",
                language="en",
                aspects=["service quality"],
                sentiment="neutral",
                needs_review=True,
                review_reason="weak_support",
                support_type="near_exact",
                confidence=0.53,
            ),
        ]
        recovered_rows, stats = _strict_topup_recovery(
            train_rows=current_rows,
            candidate_rows=candidates,
            mode="strict_topup",
            target_min_rows=2,
            confidence_threshold=0.58,
            stage_b_confidence_threshold=0.54,
            stage_c_confidence_threshold=0.52,
            staged_recovery=True,
            allow_weak_support_in_stage_c=True,
            accepted_support_types=("exact", "near_exact", "gold"),
            candidate_aspects_by_domain={"restaurant": ["food", "service"]},
            seed=42,
        )
        self.assertEqual(len(recovered_rows), 2)
        self.assertEqual(stats["topup_rows_added"], 1)
        self.assertEqual(stats["size_recovery_stage"], "C")

    def test_train_domain_leakage_filter_removes_cross_domain_rows(self) -> None:
        rows = [
            _row("r1", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
            _row("r2", domain="restaurant", language="en", aspects=["performance"], sentiment="positive"),
            _row("l1", domain="laptop", language="en", aspects=["performance"], sentiment="positive"),
            _row("l2", domain="laptop", language="en", aspects=["service quality"], sentiment="negative"),
        ]
        kept, stats = _strict_train_domain_leakage_filter(
            rows,
            candidate_aspects_by_domain={
                "restaurant": ["food", "service"],
                "laptop": ["battery", "screen", "performance"],
            },
        )
        self.assertEqual({row["id"] for row in kept}, {"r1", "l1"})
        self.assertEqual(stats["train_domain_leakage_filter_removed_rows"], 2)
        self.assertEqual(stats["train_domain_leakage_filter_removed_aspect_instances"], 2)

    def test_train_domain_leakage_metrics_report_zero_after_filter(self) -> None:
        rows = [
            _row("r1", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive"),
            _row("l1", domain="laptop", language="en", aspects=["performance"], sentiment="positive"),
        ]
        metrics = _train_domain_leakage_metrics(
            rows,
            candidate_aspects_by_domain={
                "restaurant": ["food", "service"],
                "laptop": ["battery", "screen", "performance"],
            },
        )
        self.assertEqual(metrics["train_domain_leakage_rows"], 0)
        self.assertEqual(metrics["train_domain_leakage_row_rate"], 0.0)

    def test_strict_quality_metrics_capture_contamination_and_multi_aspect(self) -> None:
        clean = _row("c1", domain="restaurant", language="en", aspects=["food quality", "service quality"], sentiment="positive", needs_review=False, support_type="near_exact")
        contaminated = _row("x1", domain="restaurant", language="en", aspects=["performance"], sentiment="neutral", needs_review=False, support_type="exact")
        contaminated["implicit"]["spans"] = [{"support_type": "exact", "label_type": "explicit", "leakage_flags": ["explicit_keyword_surface_leakage"]}]
        contaminated["implicit"]["implicit_quality_tier"] = "strict_pass"
        metrics = _strict_quality_metrics([clean, contaminated], challenge_macro_f1=1.0)
        self.assertGreater(metrics["explicit_in_implicit_rate"], 0.0)
        self.assertGreater(metrics["boundary_false_positive_count"], 0)
        self.assertGreater(metrics["multi_aspect_ratio"], 0.0)
        self.assertEqual(metrics["challenge_macro_f1"], 1.0)

    def test_strict_row_passes_rejects_explicit_span(self) -> None:
        row = _row("p1", domain="restaurant", language="en", aspects=["food quality"], sentiment="positive", needs_review=False, support_type="near_exact")
        row["implicit"]["implicit_quality_tier"] = "strict_pass"
        self.assertTrue(_strict_row_passes(row))
        row["implicit"]["spans"] = [{"support_type": "exact", "label_type": "explicit"}]
        self.assertFalse(_strict_row_passes(row))


if __name__ == "__main__":
    unittest.main()
