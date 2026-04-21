from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class RowScoringTests(unittest.TestCase):
    def test_scoring_separates_quality_usefulness_and_redundancy(self) -> None:
        from row_scoring import promotion_scoring

        train_rows = [
            {"domain": "telecom", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
            {"domain": "electronics", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
            {"domain": "electronics", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
        ]
        row = {
            "domain": "telecom",
            "sentiment": "negative",
            "abstain_acceptable": True,
            "novel_acceptable": False,
            "implicit": {
                "aspects": ["battery", "screen"],
                "dominant_sentiment": "negative",
                "hardness_tier": "H3",
                "review_reason": "low_confidence",
                "spans": [{"confidence": 0.78}],
                "aspect_confidence": {"battery": 0.78},
            },
        }

        scores = promotion_scoring(row, train_rows=train_rows, core_benchmark_domains={"telecom", "electronics"})

        self.assertGreater(scores["quality_score"], 0.0)
        self.assertGreater(scores["usefulness_score"], 0.0)
        self.assertGreater(scores["redundancy_score"], 0.0)
        self.assertGreater(scores["priority_score"], 0.0)

    def test_usefulness_beats_redundancy_when_row_fills_shortage(self) -> None:
        from row_scoring import promotion_scoring

        train_rows = [
            {"domain": "electronics", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
            {"domain": "electronics", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
            {"domain": "electronics", "implicit": {"dominant_sentiment": "neutral", "aspects": ["battery"]}},
        ]
        useful_row = {
            "domain": "telecom",
            "sentiment": "negative",
            "abstain_acceptable": True,
            "novel_acceptable": True,
            "implicit": {
                "aspects": ["network"],
                "dominant_sentiment": "negative",
                "hardness_tier": "H2",
                "review_reason": "weak_support",
                "spans": [{"confidence": 0.62}],
                "aspect_confidence": {"network": 0.62},
            },
        }
        redundant_row = {
            "domain": "electronics",
            "sentiment": "neutral",
            "abstain_acceptable": False,
            "novel_acceptable": False,
            "implicit": {
                "aspects": ["battery"],
                "dominant_sentiment": "neutral",
                "hardness_tier": "H1",
                "review_reason": "implicit_not_ready",
                "spans": [{"confidence": 0.95}],
                "aspect_confidence": {"battery": 0.95},
            },
        }

        useful_scores = promotion_scoring(useful_row, train_rows=train_rows, core_benchmark_domains={"telecom", "electronics"})
        redundant_scores = promotion_scoring(redundant_row, train_rows=train_rows, core_benchmark_domains={"telecom", "electronics"})

        self.assertGreater(useful_scores["priority_score"], redundant_scores["priority_score"])


if __name__ == "__main__":
    unittest.main()
