from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class SyntheticGenerationTests(unittest.TestCase):
    def test_synthetic_rows_carry_provenance_and_role(self) -> None:
        from synthetic_generation import generate_synthetic_multidomain

        accepted, rejected, audit = generate_synthetic_multidomain(domains=["telecom"], samples_per_domain=4)

        self.assertGreater(len(accepted), 0)
        self.assertIn("paraphrase_bank_sizes", audit)
        self.assertIn("generation_source", accepted[0])
        self.assertIn("generator_policy", accepted[0])
        self.assertIn("intended_role", accepted[0])
        self.assertIn("evidence_span", accepted[0])
        self.assertGreaterEqual(audit["accepted_total"], 1)
        self.assertEqual(audit["target_domains"], ["telecom"])

    def test_duplicate_synthetic_rows_are_rejected(self) -> None:
        from synthetic_generation import generate_synthetic_multidomain

        accepted, rejected, audit = generate_synthetic_multidomain(domains=["telecom"], samples_per_domain=2, sentiment_mix={"positive": 1.0, "negative": 0.0, "neutral": 0.0})

        self.assertGreaterEqual(audit["rejected_total"], 0)
        self.assertTrue(all("rejection_reasons" in row for row in rejected))
        self.assertTrue(all(row["target_aspect"] in {"connectivity", "service_speed", "value"} for row in accepted + rejected))


if __name__ == "__main__":
    unittest.main()
