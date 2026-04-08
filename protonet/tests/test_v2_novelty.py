import unittest

from protonet.code.calibrate_novelty import calibrate_thresholds
from protonet.code.runtime_infer import ProtonetRuntime


class _Cfg:
    selective_alpha = 0.8
    selective_beta = 0.2
    selective_gamma = 0.1
    selective_delta = 0.1
    abstain_threshold = 0.4
    multi_label_margin = 0.05
    novelty_known_threshold = 0.3
    novelty_novel_threshold = 0.7


class _RuntimeStub(ProtonetRuntime):
    def __init__(self, rows):
        self.cfg = _Cfg()
        self.novelty_calibration = {"T_known": 0.3, "T_novel": 0.7}
        self._rows = rows

    def score_text(self, review_text: str, evidence_text: str, domain: str | None = None):  # noqa: ARG002
        return list(self._rows)


class V2NoveltyTests(unittest.TestCase):
    def test_known_branch_routes_known(self) -> None:
        runtime = _RuntimeStub(
            [
                {"aspect": "battery", "sentiment": "negative", "confidence": 0.9, "raw_score": 4.0, "min_distance_sq": 0.05},
                {"aspect": "screen", "sentiment": "positive", "confidence": 0.1, "raw_score": 1.0, "min_distance_sq": 0.05},
            ]
        )
        out = runtime.score_text_selective("Battery is weak", "Battery is weak", "electronics")
        self.assertEqual(out["decision_band"], "known")
        self.assertEqual(out["decision"], "single_label")
        self.assertFalse(out["abstain"])
        self.assertEqual(out["accepted_predictions"][0]["routing"], "known")

    def test_boundary_band_abstains(self) -> None:
        runtime = _RuntimeStub(
            [
                {"aspect": "battery", "sentiment": "negative", "confidence": 0.8, "raw_score": 3.0, "min_distance_sq": 1.0},
                {"aspect": "screen", "sentiment": "neutral", "confidence": 0.2, "raw_score": 1.0, "min_distance_sq": 1.0},
            ]
        )
        out = runtime.score_text_selective("Mixed feedback", "Mixed feedback", "electronics")
        self.assertEqual(out["decision_band"], "boundary")
        self.assertEqual(out["decision"], "abstain")
        self.assertTrue(out["abstain"])

    def test_novel_band_routes_with_cluster(self) -> None:
        runtime = _RuntimeStub(
            [
                {"aspect": "packaging-damage", "sentiment": "negative", "confidence": 0.6, "raw_score": 2.0, "min_distance_sq": 9.0},
                {"aspect": "battery", "sentiment": "negative", "confidence": 0.4, "raw_score": 1.0, "min_distance_sq": 9.0},
            ]
        )
        out = runtime.score_text_selective(
            "Box arrived crushed with torn corners",
            "Box arrived crushed with torn corners",
            "electronics",
        )
        self.assertEqual(out["decision_band"], "novel")
        self.assertEqual(out["decision"], "novel")
        self.assertFalse(out["abstain"])
        self.assertTrue(out["novel_candidates"])
        candidate = out["novel_candidates"][0]
        self.assertTrue(str(candidate["novel_cluster_id"]).startswith("novel_"))
        self.assertTrue(candidate["novel_alias"])
        accepted = out["accepted_predictions"][0]
        self.assertEqual(accepted["routing"], "novel")
        self.assertEqual(accepted["novel_cluster_id"], candidate["novel_cluster_id"])

    def test_threshold_calibration_is_reproducible(self) -> None:
        rows = [
            {"novelty_score": 0.1, "novel_acceptable": False, "true_label": "battery|negative", "pred_label": "battery|negative", "confidence": 0.9},
            {"novelty_score": 0.2, "novel_acceptable": False, "true_label": "screen|positive", "pred_label": "screen|positive", "confidence": 0.8},
            {"novelty_score": 0.7, "novel_acceptable": True, "true_label": "novel", "pred_label": "battery|negative", "confidence": 0.5},
            {"novelty_score": 0.85, "novel_acceptable": True, "true_label": "novel", "pred_label": "screen|positive", "confidence": 0.4},
        ]
        first = calibrate_thresholds(rows)
        second = calibrate_thresholds(rows)
        self.assertEqual(first, second)
        self.assertLess(first["thresholds"]["T_known"], first["thresholds"]["T_novel"])
        self.assertTrue(first["validation_snapshot_hash"])

    def test_threshold_calibration_skips_without_novel(self) -> None:
        rows = [
            {"novelty_score": 0.1, "novel_acceptable": False, "true_label": "battery|negative", "pred_label": "battery|negative", "confidence": 0.9},
            {"novelty_score": 0.2, "novel_acceptable": False, "true_label": "screen|positive", "pred_label": "screen|positive", "confidence": 0.8},
        ]
        result = calibrate_thresholds(rows)
        self.assertTrue(result["not_applicable"])
        self.assertIn("warning", result)

    def test_runtime_returns_routing_and_novelty_decomposition(self) -> None:
        runtime = _RuntimeStub(
            [
                {"aspect": "packaging-damage", "sentiment": "negative", "confidence": 0.6, "raw_score": 2.0, "min_distance_sq": 9.0, "energy": 2.0},
                {"aspect": "battery", "sentiment": "negative", "confidence": 0.4, "raw_score": 1.0, "min_distance_sq": 9.0, "energy": 2.0},
            ]
        )
        out = runtime.score_text_selective("Box arrived crushed", "Box arrived crushed", "electronics")
        self.assertIn("routing", out)
        self.assertIn("novelty_decomposition", out)
        self.assertIn("distance_score", out["novelty_decomposition"])
        self.assertIn("energy_score", out["novelty_decomposition"])


if __name__ == "__main__":
    unittest.main()
