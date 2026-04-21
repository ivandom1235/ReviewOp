from __future__ import annotations

import unittest
from pydantic import ValidationError
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

class V7RowContractsTests(unittest.TestCase):
    def test_raw_loaded_requires_review_text(self) -> None:
        from row_contracts import RawLoaded
        with self.assertRaises(ValidationError):
            RawLoaded(source_id="S1")

    def test_prepared_requires_domain_and_group(self) -> None:
        from row_contracts import Prepared
        # Should fail because domain is missing
        with self.assertRaises(ValidationError):
            Prepared(row_id="R1", review_text="Great", group_id="G1")
            
    def test_grounded_requires_spans(self) -> None:
        from row_contracts import Grounded
        # Should fail because interpretations must be grounded (non-empty spans)
        with self.assertRaises(ValidationError):
            Grounded(
                row_id="R1", 
                review_text="Great", 
                domain="restaurant", 
                group_id="G1",
                interpretations=[{"aspect": "service", "sentiment": "pos"}] # Missing evidence_span
            )

    def test_benchmark_instance_has_strict_structure(self) -> None:
        from row_contracts import BenchmarkInstance
        # Test valid creation
        instance = BenchmarkInstance(
            instance_id="BI-1",
            review_text="Slow network",
            domain="telecom",
            group_id="T-GROUP-1",
            gold_interpretations=[
                {
                    "aspect": "connectivity",
                    "sentiment": "neg",
                    "evidence_span": "slow",
                    "interpretation_type": "implicit"
                }
            ],
            hardness_tier="H2",
            is_ambiguous=False
        )
        self.assertEqual(instance.domain, "telecom")

    def test_abstain_logic_in_contracts(self) -> None:
        from row_contracts import QualityScored, GroundedInterpretation
        # Validation for abstain reasons
        scored = QualityScored(
            row_id="R2",
            review_text="Maybe it works",
            domain="electronics",
            group_id="E1",
            interpretations=[
                GroundedInterpretation(
                    aspect="quality", 
                    sentiment="neutral", 
                    evidence_span="works", 
                    confidence=0.5
                )
            ],
            quality_score=0.4,
            abstain_acceptable=True,
            abstain_reason="insufficient_evidence"
        )
        self.assertEqual(scored.abstain_reason, "insufficient_evidence")

    def test_bucket_assigned_validation(self) -> None:
        from row_contracts import BucketAssigned, GroundedInterpretation
        # Test invalid bucket
        with self.assertRaises(ValidationError):
            BucketAssigned(
                row_id="R3",
                review_text="Test",
                domain="restaurant",
                group_id="G1",
                interpretations=[GroundedInterpretation(aspect="a", sentiment="p", evidence_span="e")],
                quality_score=1.0,
                bucket="invalid_bucket" # Not in Literal
            )

    def test_split_assigned_enforces_splits(self) -> None:
        from row_contracts import SplitAssigned, GroundedInterpretation
        with self.assertRaises(ValidationError):
            SplitAssigned(
                row_id="R4",
                review_text="Test",
                domain="restaurant",
                group_id="G1",
                interpretations=[GroundedInterpretation(aspect="a", sentiment="p", evidence_span="e")],
                quality_score=1.0,
                bucket="train_keep",
                split="invalid_split" # Not in Literal
            )

if __name__ == "__main__":
    unittest.main()
