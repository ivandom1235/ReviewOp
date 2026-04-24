from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_builder.benchmark.ambiguity import compute_ambiguity_score
from dataset_builder.benchmark.hardness_scorer import score_row_hardness
from dataset_builder.benchmark.novelty import assess_novelty
from dataset_builder.canonical.canonicalizer import canonicalize_interpretation
from dataset_builder.config import BuilderConfig
from dataset_builder.implicit.symptom_store import SymptomPatternStore
from dataset_builder.orchestrator.stages import InferenceStage
from dataset_builder.orchestrator.release_gate import assert_release_ready
from dataset_builder.reports.quality_report import build_quality_report
from dataset_builder.schemas.benchmark_row import BenchmarkRow
from dataset_builder.schemas.interpretation import Interpretation


def interp(**overrides) -> Interpretation:
    payload = {
        "aspect_raw": "battery",
        "aspect_canonical": "battery_life",
        "latent_family": "battery",
        "label_type": "explicit",
        "sentiment": "unknown",
        "evidence_text": "battery",
        "evidence_span": [4, 11],
        "source": "test",
        "support_type": "exact",
        "source_type": "explicit",
        "mapping_source": "exact_phrase",
    }
    payload.update(overrides)
    return Interpretation(**payload)


class InterpretationContractTests(unittest.TestCase):
    def test_rejects_invalid_source_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid source_type"):
            interp(source_type="unknown")

    def test_implicit_learned_requires_pattern_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched_pattern"):
            interp(label_type="implicit", source_type="implicit_learned", matched_pattern=None, pattern_id="p1")
        with self.assertRaisesRegex(ValueError, "pattern_id"):
            interp(label_type="implicit", source_type="implicit_learned", matched_pattern="keeps crashing", pattern_id=None)

    def test_explicit_rejects_pattern_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit interpretations cannot include pattern metadata"):
            interp(matched_pattern="battery issue", pattern_id="p1")


class SymptomStoreTests(unittest.TestCase):
    def write_store(self, rows: list[dict]) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with tmp:
            json.dump(rows, tmp)
        return Path(tmp.name)

    def test_store_rejects_missing_pattern_id(self) -> None:
        path = self.write_store([
            {"phrase": "keeps crashing", "aspect_canonical": "performance", "status": "promoted"}
        ])
        with self.assertRaisesRegex(ValueError, "pattern_id"):
            SymptomPatternStore.load(path)

    def test_store_rejects_duplicate_pattern_id(self) -> None:
        path = self.write_store([
            {"pattern_id": "p1", "phrase": "keeps crashing", "aspect_canonical": "performance", "status": "promoted"},
            {"pattern_id": "p1", "phrase": "kept crashing", "aspect_canonical": "performance", "status": "promoted"},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate pattern_id"):
            SymptomPatternStore.load(path)

    def test_exact_and_normalized_matches_return_spans(self) -> None:
        path = self.write_store([
            {
                "pattern_id": "electronics_performance_crash_001",
                "phrase": "keeps crashing",
                "aspect_canonical": "performance",
                "latent_family": "performance",
                "status": "promoted",
                "confidence": 0.87,
            }
        ])
        store = SymptomPatternStore.load(path)

        exact = store.match("This app keeps crashing on launch.", domain="electronics")
        self.assertEqual(exact[0].pattern_id, "electronics_performance_crash_001")
        self.assertEqual(exact[0].matched_text, "keeps crashing")
        self.assertEqual(exact[0].start_char, 9)
        self.assertEqual(exact[0].end_char, 23)

        normalized = store.match("This app kept crashing on launch.", domain="electronics")
        self.assertEqual(normalized[0].matched_text, "kept crashing")
        self.assertNotEqual(normalized[0].matched_text, "This app kept crashing on launch.")


class InferenceAndCanonicalizationTests(unittest.TestCase):
    def test_inference_uses_learned_pattern_id_and_span_evidence(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([
                {
                    "pattern_id": "electronics_battery_life_001",
                    "phrase": "battery doesn't last",
                    "aspect_canonical": "battery_life",
                    "latent_family": "battery",
                    "status": "promoted",
                }
            ], tmp)
            store_path = tmp.name

        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The screen is bright, but the battery doesn't last through lunch.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path))
        learned = processed.implicit_interpretations[0]

        self.assertEqual(learned.source_type, "implicit_learned")
        self.assertEqual(learned.pattern_id, "electronics_battery_life_001")
        self.assertEqual(learned.matched_pattern, "battery doesn't last")
        self.assertEqual(learned.evidence_text, processed.review_text[learned.evidence_span[0]:learned.evidence_span[1]])
        self.assertNotEqual(learned.evidence_span, [0, len(processed.review_text)])

    def test_canonicalization_preserves_learned_canonical(self) -> None:
        learned = interp(
            aspect_raw="battery doesn't last",
            aspect_canonical="battery_life",
            label_type="implicit",
            source_type="implicit_learned",
            matched_pattern="battery doesn't last",
            pattern_id="electronics_battery_life_001",
        )

        result = canonicalize_interpretation(learned, "electronics")

        self.assertEqual(result.aspect_canonical, "battery_life")


class BenchmarkQualityTests(unittest.TestCase):
    def test_novelty_has_known_boundary_and_novel_states(self) -> None:
        self.assertEqual(assess_novelty("battery_life", {"battery_life"}).status, "known")
        self.assertEqual(assess_novelty("screen_eye_strain", {"display"}, mapping_confidence=0.35, evidence_supported=True).status, "boundary")
        self.assertEqual(assess_novelty("hinge_sparks", {"display"}, mapping_confidence=0.0, evidence_supported=True).status, "novel")

    def test_h3_can_be_emitted_for_novel_ambiguous_rows(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The hinge sparks and the screen hurts my eyes.",
            gold_interpretations=[
                interp(label_type="implicit", source_type="implicit_json", aspect_raw="hinge sparks", aspect_canonical="unknown"),
                interp(label_type="implicit", source_type="implicit_json", aspect_raw="screen hurts", aspect_canonical="display"),
            ],
            ambiguity_score=0.8,
            novelty_status="novel",
        )

        self.assertGreater(compute_ambiguity_score(list(row.gold_interpretations)), 0)
        self.assertEqual(score_row_hardness(row), "H3")

    def test_quality_report_contains_diagnostic_distributions(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery doesn't last.",
            gold_interpretations=[
                interp(label_type="implicit", source_type="implicit_learned", matched_pattern="battery doesn't last", pattern_id="p1")
            ],
            novelty_status="known",
            hardness_tier="H1",
        )

        report = build_quality_report({"train": [row], "val": [], "test": []}, loaded_rows=1, processed_rows=1)

        self.assertEqual(report.source_type_distribution["implicit_learned"], 1)
        self.assertEqual(report.novelty_distribution["known"], 1)
        self.assertEqual(report.hardness_distribution["H1"], 1)
        self.assertTrue(report.accounting_valid)


class ReleaseGateTests(unittest.TestCase):
    def test_learned_run_fails_without_implicit_learned_output(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The battery is weak.",
            gold_interpretations=[interp()],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)

        with self.assertRaisesRegex(ValueError, "implicit_learned"):
            assert_release_ready(
                {"train": [row], "val": [row], "test": [row]},
                reports={"quality": report, "require_learned": True},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
                profile="diagnostic_strict"
            )

    def test_gate_fails_on_evidence_mismatch_and_accounting_mismatch(self) -> None:
        bad = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[interp(evidence_text="not present", evidence_span=[0, 7])],
        )
        report = build_quality_report({"train": [bad], "val": [bad], "test": [bad]}, loaded_rows=9, processed_rows=3, rejected_rows=6)

        with self.assertRaisesRegex(ValueError, "evidence exact-match"):
            assert_release_ready(
                {"train": [bad], "val": [bad], "test": [bad]},
                reports={"quality": report},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
                profile="diagnostic_strict"
            )

    def test_gate_fails_on_invalid_source_type_in_dict_payload(self) -> None:
        splits = {
            "train": [{"review_text": "x", "gold_interpretations": [{"source_type": "unknown", "evidence_text": "x", "evidence_span": [0, 1]}]}],
            "val": [{"review_text": "y", "gold_interpretations": [{"source_type": "explicit", "evidence_text": "y", "evidence_span": [0, 1]}]}],
            "test": [{"review_text": "z", "gold_interpretations": [{"source_type": "explicit", "evidence_text": "z", "evidence_span": [0, 1]}]}],
        }

        with self.assertRaisesRegex(ValueError, "invalid source_type"):
            assert_release_ready(
                splits,
                reports={"quality": {"total_exported": 3, "evidence": {"exact_match_rate": 1.0}, "accounting_valid": True}},
                leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            )


if __name__ == "__main__":
    unittest.main()
