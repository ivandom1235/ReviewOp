from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(ROOT))

from policy import choose_label


class PolicyTests(unittest.TestCase):
    def test_deterministic_mode_is_stable(self) -> None:
        candidates = [
            {"label": "battery_life", "probability": 0.9},
            {"label": "price", "probability": 0.1},
        ]
        a = choose_label(policy="deterministic", deterministic_label="battery_life", deterministic_confidence=0.9, candidates=candidates, temperature=0.7, seed=1, min_confidence_for_hard_map=0.8)
        b = choose_label(policy="deterministic", deterministic_label="battery_life", deterministic_confidence=0.9, candidates=candidates, temperature=0.1, seed=999, min_confidence_for_hard_map=0.8)
        self.assertEqual(a.label, b.label)
        self.assertEqual(a.confidence, b.confidence)

    def test_stochastic_mode_changes_across_seeds(self) -> None:
        candidates = [
            {"label": "battery_life", "probability": 0.51},
            {"label": "price", "probability": 0.49},
        ]
        labels = {
            choose_label(policy="stochastic", deterministic_label="battery_life", deterministic_confidence=0.51, candidates=candidates, temperature=1.0, seed=seed, min_confidence_for_hard_map=0.8).label
            for seed in range(1, 8)
        }
        self.assertGreaterEqual(len(labels), 2)

    def test_hybrid_respects_threshold(self) -> None:
        candidates = [
            {"label": "battery_life", "probability": 0.6},
            {"label": "price", "probability": 0.4},
        ]
        high = choose_label(policy="hybrid", deterministic_label="battery_life", deterministic_confidence=0.91, candidates=candidates, temperature=1.0, seed=7, min_confidence_for_hard_map=0.8)
        low = choose_label(policy="hybrid", deterministic_label="battery_life", deterministic_confidence=0.41, candidates=candidates, temperature=0.0, seed=7, min_confidence_for_hard_map=0.8)
        self.assertEqual(high.label, "battery_life")
        self.assertIn(low.label, {"battery_life", "price"})


if __name__ == "__main__":
    unittest.main()
