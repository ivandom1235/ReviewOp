from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from dataset_builder.benchmark.ambiguity import compute_ambiguity_score
from dataset_builder.benchmark.counterfactual import CounterfactualGenerator, generate_counterfactual_pairs
from dataset_builder.benchmark.hardness_scorer import score_row_hardness
from dataset_builder.benchmark.novelty import assess_novelty, aggregate_row_novelty
from dataset_builder.canonical.canonicalizer import canonicalize_interpretation
from dataset_builder.canonical.aspect_memory import AspectMemory
from dataset_builder.canonical.domain_maps import CanonicalMappingResult
from dataset_builder.canonical.open_world_fallback import mark_provisional_canonical
from dataset_builder.config import BuilderConfig
from dataset_builder.explicit.phrase_cleaning import is_noisy_label
from dataset_builder.evidence.sentence_selector import select_best_sentence
from dataset_builder.fusion.merge_candidates import merge_explicit_implicit
from dataset_builder.implicit.symptom_store import SymptomPatternStore
from dataset_builder.orchestrator.stages import InferenceStage
from dataset_builder.orchestrator.release_gate import assert_release_ready
from dataset_builder.reports.quality_report import build_quality_report
from dataset_builder.utils.row_metadata import derive_row_metadata
from dataset_builder.scripts.run_diagnostics import DiagnosticConfig, DiagnosticRunner
from dataset_builder.scripts.build_benchmark import build_arg_parser, build_config_from_args, select_working_reviews
from dataset_builder.schemas.benchmark_row import BenchmarkRow
from dataset_builder.schemas.interpretation import Interpretation
from dataset_builder.split.domain_split import choose_domain_holdout_domain, domain_holdout_split
from protonet.code.selective_decisions import combine_routing_score
from unittest.mock import patch


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
    def test_implicit_json_prefers_sentence_evidence_when_term_matches(self) -> None:
        row = BenchmarkRow(
            review_id="r-json-1",
            group_id="g1",
            domain="laptop",
            domain_family="electronics",
            review_text="The keyboard feels great. But battery life is weak and dies fast.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig())
        implicit_json = [i for i in processed.implicit_interpretations if i.source_type == "implicit_json"]
        self.assertTrue(implicit_json)
        self.assertTrue(any(i.evidence_scope in {"sentence", "phrase_window"} for i in implicit_json))
        self.assertTrue(any(i.implicit_trigger == "latent_family_match" for i in implicit_json))

    def test_sentence_selector_narrows_single_sentence_review_when_cue_exists(self) -> None:
        text = "Plain and simple, it runs great and loads fast."
        selected = select_best_sentence(text, "fast")
        self.assertNotEqual(selected, text)
        self.assertIn("fast", selected.lower())

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

    def test_canonicalization_propagates_mapping_scope_and_layers(self) -> None:
        item = interp(aspect_raw="service quick")
        with patch("dataset_builder.canonical.canonicalizer.lookup_domain_map") as mocked_lookup:
            mocked_lookup.return_value = CanonicalMappingResult(
                aspect_canonical="service_speed",
                mapping_source="anchor_modifier",
                mapping_confidence=0.85,
                mapping_scope="generic+domain_specific",
                mapping_layers=("generic", "domain_specific"),
            )
            result = canonicalize_interpretation(item, "restaurant")

        self.assertEqual(result.mapping_scope, "generic+domain_specific")
        self.assertEqual(result.mapping_layers, ("generic", "domain_specific"))

    def test_inference_emits_behavior_pattern_novel_candidate(self) -> None:
        row = BenchmarkRow(
            review_id="r-behavior-1",
            group_id="g-behavior-1",
            domain="telecom",
            domain_family="telecom",
            review_text="My calls kept dropping every few minutes during the commute.",
        )

        [processed] = InferenceStage().process([row], BuilderConfig())
        behavior = [i for i in processed.implicit_interpretations if i.source == "behavior_pattern_matcher"]

        self.assertTrue(behavior)
        self.assertEqual(behavior[0].aspect_canonical, "call_reliability")
        self.assertEqual(behavior[0].mapping_source, "open_world_candidate")

    def test_canonicalization_preserves_behavior_pattern_discoveries(self) -> None:
        behavior = interp(
            aspect_raw="calls kept dropping",
            aspect_canonical="call_reliability",
            latent_family="call_reliability",
            label_type="implicit",
            source="behavior_pattern_matcher",
            source_type="implicit_json",
            mapping_source="open_world_candidate",
            mapping_scope="open_world_candidate",
            mapping_layers=("open_world_candidate",),
            canonical_confidence=0.85,
        )

        result = canonicalize_interpretation(behavior, "telecom")

        self.assertEqual(result.aspect_canonical, "call_reliability")
        self.assertEqual(result.mapping_source, "open_world")
        self.assertEqual(result.mapping_scope, "open_world")


class BenchmarkQualityTests(unittest.TestCase):
    def test_novelty_has_known_boundary_and_novel_states(self) -> None:
        self.assertEqual(assess_novelty("battery_life", {"battery_life"}).status, "known")
        self.assertEqual(assess_novelty("screen_eye_strain", {"display"}, mapping_confidence=0.35, evidence_supported=True).status, "boundary")
        self.assertEqual(assess_novelty("hinge_sparks", {"display"}, mapping_confidence=0.8, evidence_supported=True).status, "novel")

    def test_row_novelty_requires_a_truly_novel_interpretation(self) -> None:
        row = [
            interp(
                aspect_raw="portable stand",
                aspect_canonical="portable_stand",
                mapping_source="provisional",
                mapping_scope="provisional",
                canonical_confidence=0.35,
                novelty_status="boundary",
            ),
            interp(
                aspect_raw="battery",
                aspect_canonical="battery_life",
                mapping_source="exact_phrase",
                mapping_scope="generic",
                canonical_confidence=1.0,
                novelty_status="known",
            ),
        ]

        self.assertEqual(aggregate_row_novelty(row), "boundary")

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
        self.assertIn("mapping_scope_distribution", report.__dict__)
        self.assertTrue(report.accounting_valid)

    def test_row_metadata_is_derived_from_gold_interpretations(self) -> None:
        row = BenchmarkRow(
            review_id="r-meta-1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[
                interp(
                    label_type="explicit",
                    source_type="explicit",
                    source="explicit",
                    mapping_source="exact_phrase",
                    mapping_scope="generic",
                ),
                interp(
                    label_type="implicit",
                    source_type="implicit_learned",
                    source="learned",
                    matched_pattern="battery weak",
                    pattern_id="p1",
                    mapping_source="trusted_learned",
                    mapping_scope="learned_store",
                ),
            ],
        )

        derived = derive_row_metadata(row)

        self.assertEqual(derived.row_source_type, "hybrid")
        self.assertEqual(derived.row_mapping_scope, "mixed")
        self.assertEqual(derived.row_mapping_sources, ("exact_phrase", "trusted_learned"))
        self.assertEqual(derived.source_type, "hybrid")
        self.assertEqual(derived.mapping_scope, "mixed")
        self.assertEqual(derived.mapping_source, "exact_phrase|trusted_learned")

    def test_row_metadata_marks_empty_abstain_rows_without_unknowns(self) -> None:
        row = BenchmarkRow(
            review_id="r-meta-abstain",
            group_id="g1",
            domain="shopping",
            domain_family="shopping",
            review_text="Something about the app was frustrating.",
            gold_interpretations=[],
            abstain_acceptable=True,
            abstain_reason_gold=("insufficient_aspect_evidence", "vague_review"),
        )

        derived = derive_row_metadata(row)

        self.assertEqual(derived.row_source_type, "abstain")
        self.assertEqual(derived.row_mapping_scope, "abstain")
        self.assertEqual(derived.row_mapping_sources, ("abstain",))

    def test_quality_report_counts_top_level_row_metadata_unknown_fields(self) -> None:
        row = BenchmarkRow(
            review_id="r-meta-2",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[interp()],
        )

        report = build_quality_report({"train": [row], "val": [], "test": []}, loaded_rows=1, processed_rows=1)

        self.assertEqual(report.canonicalization["row_metadata_unknown_count"], 1)


class ReleaseGateTests(unittest.TestCase):
    def test_research_default_thresholds_for_provisional_and_anchor_modifier(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
            gold_interpretations=[
                interp(
                    mapping_source="provisional",
                    source_type="explicit",
                    label_type="explicit",
                    mapping_scope="provisional",
                    evidence_text="Battery",
                    evidence_span=[0, 7],
                )
            ],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)
        result = assert_release_ready(
            {"train": [row], "val": [row], "test": [row]},
            reports={"quality": report},
            leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            profile="research_default",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("provisional" in f for f in result["failures"]))

    def test_gate_fails_when_mapping_scope_unknown_exists(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="The battery is weak.",
            gold_interpretations=[interp(mapping_scope="unknown")],
        )
        report = build_quality_report({"train": [row], "val": [row], "test": [row]}, loaded_rows=3, processed_rows=3)
        result = assert_release_ready(
            {"train": [row], "val": [row], "test": [row]},
            reports={"quality": report},
            leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            profile="research_default",
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("mapping_scope" in f for f in result["failures"]))

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

    def test_quality_report_uses_runtime_row_rejection_reason_counts(self) -> None:
        report = build_quality_report(
            {"train": [], "val": [], "test": []},
            loaded_rows=10,
            processed_rows=8,
            rejected_rows=2,
            runtime_reason_counts={
                "duplicate_text_dropped": 1,
                "empty_gold_after_canonicalization": 1,
            },
        )

        self.assertEqual(
            report.row_rejection_reason_counts,
            {
                "duplicate_text_dropped": 1,
                "empty_gold_after_canonicalization": 1,
            },
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

    def test_release_gate_falls_back_to_row_rejection_reason_counts_in_strict_profile(self) -> None:
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="Battery is weak.",
        )
        quality = {
            "accounting_valid": True,
            "total_exported": 3,
            "rejected_rows": 1,
            "reason_counts": {},
            "row_rejection_reason_counts": {"empty_gold_after_canonicalization": 1},
            "evidence": {
                "exact_match_rate": 1.0,
                "full_review_evidence_rate": 0.0,
                "matched_term_in_evidence_rate": 1.0,
            },
            "canonicalization": {
                "unknown_rate": 0.0,
                "mapping_scope_unknown_count": 0,
                "row_metadata_unknown_count": 0,
                "provisional_rate": 0.0,
                "anchor_modifier_count": 1,
            },
            "novelty_distribution": {"novel": 1},
            "mapping_source_distribution": {"exact_phrase": 1},
            "source_type_distribution": {"explicit": 1},
        }

        result = assert_release_ready(
            {"train": [row], "val": [row], "test": [row]},
            reports={"quality": quality},
            leakage={"grouped_leakage": 0, "exact_text_leakage": 0},
            profile="diagnostic_strict",
        )

        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["failures"])
        self.assertFalse(result["warnings"])


class BuildBenchmarkCliTests(unittest.TestCase):
    def test_cli_exposes_new_flags_and_wires_builder_config(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "dataset_builder/input",
            "--strict",
            "--domain-mode",
            "generic_only",
            "--domain-holdout-domain",
            "electronics",
            "--aspect-memory",
            "dataset_builder/config/aspect_memory/memory_v001.json",
            "--aspect-memory-bootstrap",
        ])
        cfg = build_config_from_args(args, [Path("dataset_builder/input/reviews.jsonl")])
        self.assertTrue(cfg.strict)
        self.assertEqual(cfg.domain_mode, "generic_only")
        self.assertEqual(cfg.domain_holdout_domain, "electronics")
        self.assertEqual(Path(cfg.aspect_memory_path), Path("dataset_builder/config/aspect_memory/memory_v001.json"))
        self.assertTrue(cfg.aspect_memory_bootstrap)

    def test_select_working_reviews_prioritizes_fixture_rows(self) -> None:
        from dataset_builder.schemas.raw_review import RawReview

        rows = [
            RawReview(review_id="a", group_id="a", domain="restaurant", domain_family="restaurant", text="regular a", metadata={}),
            RawReview(review_id="b", group_id="b", domain="restaurant", domain_family="restaurant", text="fixture b", metadata={"fixture_priority": "must_include"}),
            RawReview(review_id="c", group_id="c", domain="restaurant", domain_family="restaurant", text="fixture c", metadata={"fixture_priority": "must_include"}),
            RawReview(review_id="d", group_id="d", domain="restaurant", domain_family="restaurant", text="regular d", metadata={}),
        ]

        selected = select_working_reviews(rows, BuilderConfig(sample_size=3, random_seed=42))
        selected_ids = {row.review_id for row in selected}

        self.assertIn("b", selected_ids)
        self.assertIn("c", selected_ids)


class DomainHoldoutAndCounterfactualTests(unittest.TestCase):
    def test_domain_holdout_split_marks_holdout_domain(self) -> None:
        rows = [
            BenchmarkRow(review_id="r1", group_id="g1", domain="electronics", domain_family="electronics", review_text="Battery is weak."),
            BenchmarkRow(review_id="r2", group_id="g2", domain="electronics", domain_family="electronics", review_text="Screen is bright."),
            BenchmarkRow(review_id="r3", group_id="g3", domain="restaurant", domain_family="dining", review_text="Service was slow."),
            BenchmarkRow(review_id="r4", group_id="g4", domain="hotel", domain_family="hospitality", review_text="The room was clean."),
        ]

        holdout = choose_domain_holdout_domain(rows)
        self.assertEqual(holdout, "electronics")

        split = domain_holdout_split(rows, holdout)
        self.assertEqual(len(split["test"]), 2)
        self.assertEqual(len(split["train"]), 1)
        self.assertGreaterEqual(len(split["val"]), 1)
        self.assertTrue(all(getattr(row, "split_protocol", {}).get("domain_holdout") == "test" for row in split["test"]))
        self.assertTrue(all(getattr(row, "split_protocol", {}).get("domain_holdout") == "train" for row in split["train"]))

    def test_counterfactual_generator_emits_paired_rows(self) -> None:
        rows = [
            BenchmarkRow(review_id="r1", group_id="g1", domain="restaurant", domain_family="dining", review_text="Food was cold."),
            BenchmarkRow(review_id="r2", group_id="g2", domain="restaurant", domain_family="dining", review_text="The food was good."),
        ]
        cf_res = generate_counterfactual_pairs(rows, max_pairs=10)
        pairs = cf_res["pairs"]

        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(all(pair["counterfactual_group_id"].startswith("cf_") for pair in pairs))
        self.assertTrue(any(pair["rewrite_type"] == "aspect_swap" for pair in pairs))
        self.assertTrue(any(pair["counterfactual_text"] != pair["original_text"] for pair in pairs))

    def test_counterfactual_generator_prefers_aspect_swaps_before_sentiment_flips(self) -> None:
        rows = [
            BenchmarkRow(review_id="r1", group_id="g1", domain="restaurant", domain_family="dining", review_text="The food was good."),
            BenchmarkRow(review_id="r2", group_id="g2", domain="restaurant", domain_family="dining", review_text="Food was cold."),
        ]

        cf_res = generate_counterfactual_pairs(rows, max_pairs=1, min_aspect_swaps=1)
        pairs = cf_res["pairs"]

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["rewrite_type"], "aspect_swap")
        self.assertEqual(pairs[0]["counterfactual_text"].rstrip("."), "staff was cold")

    def test_counterfactual_generator_synthesizes_aspect_swaps_when_rows_do_not_match_templates(self) -> None:
        rows = [
            BenchmarkRow(review_id="r1", group_id="g1", domain="restaurant", domain_family="dining", review_text="The dessert was fine."),
            BenchmarkRow(review_id="r2", group_id="g2", domain="restaurant", domain_family="dining", review_text="The meal was ordinary."),
        ]

        cf_res = generate_counterfactual_pairs(rows, max_pairs=2, min_aspect_swaps=1)
        pairs = cf_res["pairs"]

        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(any(pair["rewrite_type"] == "aspect_swap" for pair in pairs))
        self.assertTrue(any(pair["original_review_id"].startswith("synthetic_aspect_swap_") for pair in pairs))

    def test_counterfactual_generation_reports_source_breakdown(self) -> None:
        rows = [
            BenchmarkRow(review_id="r1", group_id="g1", domain="restaurant", domain_family="dining", review_text="The food was cold."),
            BenchmarkRow(review_id="r2", group_id="g2", domain="restaurant", domain_family="dining", review_text="The dessert was ordinary."),
        ]

        cf_res = generate_counterfactual_pairs(rows, max_pairs=3, min_aspect_swaps=2)

        self.assertIn("source_counts", cf_res["stats"])
        self.assertIn("natural_match", cf_res["stats"]["source_counts"])
        self.assertIn("synthetic_seed", cf_res["stats"]["source_counts"])
        self.assertEqual(sum(cf_res["stats"]["source_counts"].values()), cf_res["stats"]["exported"])
        self.assertTrue(all("counterfactual_source" in pair for pair in cf_res["pairs"]))
        self.assertIn("natural_aspect_swap_count", cf_res["stats"])
        self.assertIn("synthetic_aspect_swap_count", cf_res["stats"])
        self.assertGreaterEqual(cf_res["stats"]["natural_aspect_swap_count"], 1)

    def test_counterfactual_generator_blocks_or_repairs_known_bad_phrases(self) -> None:
        generator = CounterfactualGenerator()

        blocked = generator.rewrite("Happy hour at this place is always packed.")
        repaired = generator.rewrite("I highly recommend Cafe St. Bart's.")

        self.assertIsNone(blocked)
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired[0], "I would avoid Cafe St. Bart's.")

    def test_anchor_modifier_survives_merge_against_exact_phrase_peer(self) -> None:
        exact = interp(
            aspect_raw="The food was delicious",
            aspect_canonical="food_quality",
            mapping_source="exact_phrase",
            evidence_text="The food was delicious",
            evidence_span=[0, 22],
        )
        anchor = interp(
            aspect_raw="The food was delicious",
            aspect_canonical="food_quality",
            mapping_source="anchor_modifier",
            aspect_anchor="food",
            modifier_terms=("delicious",),
            anchor_source="noun_chunk_root",
            evidence_text="The food was delicious",
            evidence_span=[0, 22],
        )

        merged = merge_explicit_implicit([exact], [anchor])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].mapping_source, "anchor_modifier")

    def test_anchor_modifier_survives_merge_against_higher_confidence_exact_phrase_peer(self) -> None:
        exact = interp(
            aspect_raw="The food was delicious",
            aspect_canonical="food_quality",
            canonical_confidence=1.0,
            mapping_source="exact_phrase",
            evidence_text="The food was delicious",
            evidence_span=[0, 22],
        )
        anchor = interp(
            aspect_raw="The food was delicious",
            aspect_canonical="food_quality",
            canonical_confidence=0.85,
            mapping_source="anchor_modifier",
            aspect_anchor="food",
            modifier_terms=("delicious",),
            anchor_source="noun_chunk_root",
            evidence_text="The food was delicious",
            evidence_span=[0, 22],
        )

        merged = merge_explicit_implicit([exact], [anchor])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].mapping_source, "anchor_modifier")


class MappingAndUnseenDomainTests(unittest.TestCase):
    def test_anchor_modifier_expected_cases(self) -> None:
        cases = [
            ("laptop", "screen", ("dim",), "display"),
            ("laptop", "battery", ("weak",), "battery_life"),
            ("laptop", "laptop", ("portable",), "portability"),
            ("restaurant", "service", ("quick",), "service_speed"),
            ("restaurant", "food", ("fresh",), "food_quality"),
            ("laptop", "speaker", ("loud",), "audio"),
        ]
        for domain, anchor, mods, expected in cases:
            out = canonicalize_interpretation(
                interp(
                    aspect_raw=f"{anchor} {' '.join(mods)}",
                    aspect_anchor=anchor,
                    modifier_terms=mods,
                    anchor_source="test",
                ),
                domain,
            )
            self.assertEqual(out.aspect_canonical, expected)

    def test_unseen_domain_generic_fallback_no_crash(self) -> None:
        row = BenchmarkRow(
            review_id="fashion_001",
            group_id="g_fashion_1",
            domain="fashion",
            domain_family="unknown",
            review_text="The stitching came apart after one wash, but the fabric felt soft.",
        )
        [processed] = InferenceStage().process([row], BuilderConfig())
        self.assertIsNotNone(processed)

    def test_canonicalization_attaches_generic_parent_for_food_aspects(self) -> None:
        out = canonicalize_interpretation(
            interp(
                aspect_raw="pizza",
                aspect_canonical="unknown",
                evidence_text="pizza",
                evidence_span=[0, 5],
                mapping_source="provisional",
            ),
            "restaurant",
        )
        self.assertNotEqual(out.aspect_canonical, "unknown")

    def test_canonicalization_preserves_anchor_modifier_for_modifier_phrase(self) -> None:
        out = canonicalize_interpretation(
            interp(
                aspect_raw="The food was delicious",
                aspect_canonical="unknown",
                aspect_anchor="food",
                modifier_terms=("delicious",),
                anchor_source="noun_chunk_root",
                evidence_text="The food was delicious",
                evidence_span=[0, 22],
                mapping_source="unmapped",
            ),
            "restaurant",
        )
        self.assertEqual(out.mapping_source, "anchor_modifier")

    def test_canonicalization_promotes_modifier_tokens_over_token_fallback(self) -> None:
        out = canonicalize_interpretation(
            interp(
                aspect_raw="cheap lunch",
                aspect_canonical="unknown",
                aspect_anchor="lunch",
                modifier_terms=("cheap",),
                anchor_source="noun_chunk_root",
                evidence_text="cheap lunch",
                evidence_span=[0, 11],
                mapping_source="unmapped",
            ),
            "restaurant",
        )
        self.assertEqual(out.mapping_source, "anchor_modifier")

    def test_memory_candidate_is_rescued_when_row_would_otherwise_be_empty(self) -> None:
        from dataset_builder.orchestrator.stages import CanonicalizationStage

        root = Path("dataset_builder/output/_tmp_memory_rescue_test")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        out = root / "out"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(root / "memory.json"),
        )
        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="restaurant",
            domain_family="dining",
            review_text="The personal sizes were surprisingly generous.",
            gold_interpretations=(
                interp(
                    aspect_raw="personal sizes",
                    aspect_canonical="unknown",
                    aspect_anchor="sizes",
                    modifier_terms=("personal",),
                    anchor_source="noun_chunk_root",
                    evidence_text="personal sizes",
                    evidence_span=[0, 14],
                    mapping_source="unmapped",
                ),
            ),
        )
        try:
            [processed] = CanonicalizationStage().process([row], cfg)
            self.assertTrue(processed.gold_interpretations)
            self.assertNotEqual(processed.gold_interpretations[0].aspect_canonical, "unknown")
            self.assertEqual(processed.gold_interpretations[0].mapping_source, "open_world_candidate")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class DiagnosticsAndMemoryTddTests(unittest.TestCase):
    def test_should_enter_aspect_memory_filters_sentiment_only_and_accepts_behavioral_patterns(self) -> None:
        from dataset_builder.orchestrator.stages import _should_enter_aspect_memory

        row = BenchmarkRow(
            review_id="r1",
            group_id="g1",
            domain="electronics",
            domain_family="electronics",
            review_text="My calls kept dropping every few minutes.",
        )

        rejected = interp(
            aspect_raw="great battery life",
            aspect_canonical="great_battery_life",
            mapping_source="open_world_candidate",
            evidence_text="great battery life",
            evidence_span=[0, 18],
        )
        accepted = interp(
            aspect_raw="calls kept dropping",
            aspect_canonical="call_reliability",
            mapping_source="provisional",
            evidence_text="My calls kept dropping every few minutes.",
            evidence_span=[3, 42],
        )

        rejected_ok, rejected_reason = _should_enter_aspect_memory(rejected, row)
        accepted_ok, accepted_trigger = _should_enter_aspect_memory(accepted, row)

        self.assertFalse(rejected_ok)
        self.assertEqual(rejected_reason, "")
        self.assertTrue(accepted_ok)
        self.assertEqual(accepted_trigger, "calls kept dropping")

    def test_mark_provisional_canonical_rejects_noisy_sentiment_labels(self) -> None:
        self.assertEqual(mark_provisional_canonical("great evening"), "")
        self.assertEqual(mark_provisional_canonical("good product"), "")
        self.assertEqual(mark_provisional_canonical("excellent proprietary software"), "")
        self.assertEqual(mark_provisional_canonical("call reliability"), "call_reliability")

    def test_diagnostics_include_unseen_domain_flag_filters_fixture_loading(self) -> None:
        root = Path("dataset_builder/output/_tmp_diag_test")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        fixtures = root / "fixtures"
        fixtures.mkdir(parents=True)
        (fixtures / "regular.jsonl").write_text(
            json.dumps({"id": "r1", "fixture_type": "regular", "review_text": "ok"}) + "\n",
            encoding="utf-8",
        )
        (fixtures / "unseen_domain_fashion.jsonl").write_text(
            json.dumps({"id": "u1", "fixture_type": "unseen_domain", "review_text": "ok"}) + "\n",
            encoding="utf-8",
        )
        exp = root / "exp.yaml"
        exp.write_text("{}", encoding="utf-8")
        cfg = DiagnosticConfig(fixtures, exp, root / "out")
        runner = DiagnosticRunner(cfg)
        rows = runner.load_fixtures()
        self.assertEqual(len(rows), 1)

    def test_metrics_summary_contains_aspect_memory_block(self) -> None:
        from dataset_builder.orchestrator.pipeline import run_builder_pipeline
        from dataset_builder.schemas.raw_review import RawReview

        root = Path("dataset_builder/output/_tmp_metrics_test")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        out = root / "out"
        memory = root / "memory.json"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(memory),
        )
        raws = [
            RawReview(
                review_id="r1",
                group_id="g1",
                text="Thing works but stitching came apart and battery weak",
                domain="fashion",
                domain_family="fashion",
                source_name="test",
            )
        ]
        with patch("dataset_builder.orchestrator.pipeline.assert_release_ready") as gate:
            gate.return_value = {"status": "PASS", "failures": [], "warnings": [], "metrics": {}}
            with patch("dataset_builder.orchestrator.pipeline.ExtractionStage") as mock_ext, \
                 patch("dataset_builder.orchestrator.pipeline.InferenceStage") as mock_inf, \
                 patch("dataset_builder.orchestrator.pipeline.FusionStage") as mock_fus, \
                 patch("dataset_builder.orchestrator.pipeline.EvidenceStage") as mock_ev, \
                 patch("dataset_builder.orchestrator.pipeline.VerificationStage") as mock_ver, \
                 patch("dataset_builder.orchestrator.pipeline.PostVerificationEvidenceStage") as mock_pv, \
                 patch("dataset_builder.orchestrator.pipeline.CanonicalizationStage") as mock_can, \
                 patch("dataset_builder.orchestrator.pipeline.SentimentStage") as mock_sen, \
                 patch("dataset_builder.orchestrator.pipeline.BenchmarkStage") as mock_ben:
                for m in [mock_ext, mock_inf, mock_fus, mock_ev, mock_ver, mock_pv, mock_can, mock_sen, mock_ben]:
                    m.return_value.process.side_effect = lambda rows, _cfg: rows
                run_builder_pipeline(cfg, raw_reviews=raws)
        payload = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
        self.assertIn("aspect_memory", payload)
        for key in (
            "candidates_added",
            "promoted_matches_used",
            "candidates_promoted_this_run",
            "promoted_entries_total",
            "review_queue_count",
            "rejected_candidates_this_run",
        ):
            self.assertIn(key, payload["aspect_memory"])

    def test_controlled_aspect_memory_bootstrap_populates_review_queue(self) -> None:
        from dataset_builder.orchestrator.stages import _bootstrap_controlled_aspect_memory

        root = Path("dataset_builder/output/_tmp_aspect_memory_bootstrap")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(root / "memory.json")
            review_queue_count = _bootstrap_controlled_aspect_memory(memory, run_id="run_bootstrap_test")
            memory.save()
            memory.write_summary(root / "aspect_memory_summary.json")

            summary = json.loads((root / "aspect_memory_summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(review_queue_count, 3)
            self.assertGreaterEqual(summary["review_queue_count"], 3)
            self.assertGreaterEqual(summary["top_clusters"][0]["support_count"], 3)
            self.assertEqual(summary["bootstrap_entry_count"], 3)
            self.assertEqual(summary["organic_entry_count"], 0)
            self.assertEqual(summary["bootstrap_review_queue_count"], 3)
            self.assertEqual(summary["organic_review_queue_count"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_metrics_summary_merges_aspect_memory_summary_fields(self) -> None:
        from dataset_builder.orchestrator.pipeline import run_builder_pipeline
        from dataset_builder.schemas.raw_review import RawReview

        root = Path("dataset_builder/output/_tmp_metrics_summary_sync")
        if root.exists():
            shutil.rmtree(root)
        out = root / "out"
        memory = root / "memory.json"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(memory),
        )
        raws = [
            RawReview(
                review_id="r1",
                group_id="g1",
                text="The calls kept dropping and the screen looked fine.",
                domain="electronics",
                domain_family="electronics",
                source_name="test",
            )
        ]
        summary_payload = {
            "total_entries": 4,
            "review_queue_count": 2,
            "promoted_count": 1,
            "rejected_count": 3,
            "unknown_candidate_count": 0,
            "broad_noun_candidate_rate": 0.0,
            "evidence_pattern_candidate_rate": 0.75,
            "top_clusters": [{"cluster_id": "mem_call_reliability", "support_count": 3}],
        }

        with patch("dataset_builder.orchestrator.pipeline.assert_release_ready") as gate:
            gate.return_value = {"status": "PASS", "failures": [], "warnings": [], "metrics": {}}
            with patch("dataset_builder.orchestrator.pipeline.ExtractionStage") as mock_ext, \
                 patch("dataset_builder.orchestrator.pipeline.InferenceStage") as mock_inf, \
                 patch("dataset_builder.orchestrator.pipeline.FusionStage") as mock_fus, \
                 patch("dataset_builder.orchestrator.pipeline.EvidenceStage") as mock_ev, \
                 patch("dataset_builder.orchestrator.pipeline.VerificationStage") as mock_ver, \
                 patch("dataset_builder.orchestrator.pipeline.PostVerificationEvidenceStage") as mock_pv, \
                 patch("dataset_builder.orchestrator.pipeline.CanonicalizationStage") as mock_can, \
                 patch("dataset_builder.orchestrator.pipeline.SentimentStage") as mock_sen, \
                 patch("dataset_builder.orchestrator.pipeline.BenchmarkStage") as mock_ben:
                for m in [mock_ext, mock_inf, mock_fus, mock_ev, mock_ver, mock_pv, mock_sen, mock_ben]:
                    m.return_value.process.side_effect = lambda rows, _cfg: rows

                def write_summary(rows, _cfg):
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "aspect_memory_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")
                    return rows

                mock_can.return_value.process.side_effect = write_summary
                run_builder_pipeline(cfg, raw_reviews=raws)

        payload = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
        aspect_memory = payload["aspect_memory"]
        self.assertEqual(aspect_memory["review_queue_count"], 2)
        self.assertEqual(aspect_memory["unknown_candidate_count"], 0)
        self.assertEqual(aspect_memory["broad_noun_candidate_rate"], 0.0)
        self.assertEqual(aspect_memory["evidence_pattern_candidate_rate"], 0.75)
        self.assertIn("top_clusters", aspect_memory)

    def test_conflict_resolution_prefers_symptom_store_by_default(self) -> None:
        memory_path = Path("dataset_builder/output/_tmp_memory_conflict/memory.json")
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_payload = {
            "entries": {
                "battery weak": {
                    "cluster_id": "mem_battery_weak",
                    "aspect_raw": "battery weak",
                    "status": "promoted",
                    "validation_status": "auto_validated",
                    "support_count": 5,
                    "unique_reviews": ["r1"],
                    "domains": ["electronics"],
                    "trigger_patterns": ["battery weak"],
                    "evidence_examples": [{"evidence_text": "battery weak", "review_id": "r1", "domain": "electronics"}],
                    "suggested_aspect": "battery_health",
                    "generic_parent": "quality",
                }
            }
        }
        memory_path.write_text(json.dumps(memory_payload), encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([{"pattern_id":"p1","phrase":"battery weak","aspect_canonical":"battery_life","status":"promoted","confidence":0.9}], tmp)
            store_path = tmp.name
        row = BenchmarkRow(review_id="r1", group_id="g1", domain="electronics", domain_family="electronics", review_text="battery weak")
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path, aspect_memory_path=str(memory_path)))
        learned = [i for i in processed.implicit_interpretations if i.source_type == "implicit_learned"][0]
        self.assertEqual(learned.aspect_canonical, "battery_life")
        self.assertIn("symptom_store", learned.mapping_layers)
        self.assertIn("aspect_memory", learned.mapping_layers)
        self.assertEqual(getattr(learned, "conflict_resolution", "none"), "symptom_store_preferred")

    def test_conflict_resolution_allows_manual_validated_memory_override(self) -> None:
        memory_path = Path("dataset_builder/output/_tmp_memory_conflict2/memory.json")
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_payload = {
            "entries": {
                "battery weak": {
                    "cluster_id": "mem_battery_weak",
                    "aspect_raw": "battery weak",
                    "status": "promoted",
                    "validation_status": "manual_validated",
                    "support_count": 5,
                    "unique_reviews": ["r1"],
                    "domains": ["electronics"],
                    "trigger_patterns": ["battery weak"],
                    "evidence_examples": [{"evidence_text": "battery weak", "review_id": "r1", "domain": "electronics"}],
                    "suggested_aspect": "battery_health",
                    "generic_parent": "quality",
                    "contradiction_score": 0.0
                }
            }
        }
        memory_path.write_text(json.dumps(memory_payload), encoding="utf-8")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump([{"pattern_id":"p1","phrase":"battery weak","aspect_canonical":"battery_life","status":"promoted","confidence":0.8}], tmp)
            store_path = tmp.name
        row = BenchmarkRow(review_id="r1", group_id="g1", domain="electronics", domain_family="electronics", review_text="battery weak")
        [processed] = InferenceStage().process([row], BuilderConfig(symptom_store_path=store_path, aspect_memory_path=str(memory_path)))
        learned = [i for i in processed.implicit_interpretations if i.source_type == "implicit_learned"][0]
        self.assertEqual(learned.aspect_canonical, "battery_health")
        self.assertEqual(getattr(learned, "conflict_resolution", "none"), "manual_validated_aspect_memory_preferred")

    def test_review_queue_file_contains_required_keys(self) -> None:
        root = Path("dataset_builder/output/_tmp_review_queue")
        root.mkdir(parents=True, exist_ok=True)
        memory = AspectMemory(root / "memory.json")
        memory.add_evidence("thing", "r1", "thing broke", "fashion")
        queue_path = root / "aspect_memory_review_queue.json"
        memory.write_review_queue(queue_path)
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        self.assertIn("items", payload)
        if payload["items"]:
            item = payload["items"][0]
            for k in ("cluster_id", "aspect_raw", "suggested_aspect", "support_count", "unique_review_count", "evidence_examples", "consistency", "quality"):
                self.assertIn(k, item)

    def test_aspect_memory_promotes_review_queue_for_supported_aspect(self) -> None:
        tmp = Path("dataset_builder/output/_tmp_aspect_memory_test")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(tmp / "memory.json")
            memory.add_evidence("pizza", "r1", "the crust was soggy and undercooked", "restaurant")
            memory.add_evidence("pizza", "r2", "pizza crust stayed soggy", "restaurant")
            memory.add_evidence("pizza", "r3", "the crust came out undercooked again", "restaurant")

            entry = memory.get_entry("pizza")
            self.assertIsNotNone(entry)
            self.assertIsNone(entry.generic_parent)
            self.assertEqual(entry.generic_parent_status, "not_assigned")
            self.assertEqual(entry.status, "review_queue")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_aspect_memory_rejects_single_surface_form_clusters(self) -> None:
        tmp = Path("dataset_builder/output/_tmp_aspect_memory_single_surface")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(tmp / "memory.json")
            for idx in range(3):
                memory.add_evidence("pizza", f"r{idx + 1}", "the pizza was delicious", "restaurant")

            entry = memory.get_entry("pizza")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.status, "detected")
            self.assertEqual(len(set(entry.trigger_patterns)), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_aspect_memory_uses_evidence_phrase_instead_of_raw_aspect_fallback(self) -> None:
        tmp = Path("dataset_builder/output/_tmp_aspect_memory_trigger")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(tmp / "memory.json")
            cluster_id = memory.add_evidence("pizza", "r1", "The crust was soggy and undercooked.", "restaurant")
            entry = memory.entries.get(cluster_id)
            self.assertIsNotNone(entry)
            self.assertTrue(entry.trigger_patterns)
            self.assertNotIn("pizza", entry.trigger_patterns[0])
            self.assertTrue(any("soggy" in pattern or "undercooked" in pattern for pattern in entry.trigger_patterns))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_aspect_memory_rejects_pure_sentiment_phrases(self) -> None:
        tmp = Path("dataset_builder/output/_tmp_aspect_memory_sentiment")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(tmp / "memory.json")
            cluster_id = memory.add_evidence("evening", "r1", "great evening", "restaurant")
            self.assertEqual(cluster_id, "rejected_noise")
            self.assertFalse(memory.entries)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_aspect_memory_rejects_explicit_sentiment_phrase_with_broad_noun(self) -> None:
        tmp = Path("dataset_builder/output/_tmp_aspect_memory_sentiment_long")
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(tmp / "memory.json")
            cluster_id = memory.add_evidence("battery", "r1", "great battery life", "electronics")
            self.assertEqual(cluster_id, "rejected_noise")
            self.assertFalse(memory.entries)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_evidence_quality_scores_behavioral_text_higher_than_generic_or_sentiment_text(self) -> None:
        memory = AspectMemory(Path("dataset_builder/output/_tmp_quality_test.json"))
        self.assertGreater(memory._basic_evidence_quality("calls kept dropping"), 0.7)
        self.assertGreater(memory._basic_evidence_quality("threads came loose"), 0.7)
        self.assertLess(memory._basic_evidence_quality("great evening"), 0.5)
        self.assertLess(memory._basic_evidence_quality("pizza"), 0.3)

    def test_pipeline_populates_aspect_memory_review_queue_from_contextual_evidence(self) -> None:
        from dataset_builder.orchestrator.stages import CanonicalizationStage
        from dataset_builder.schemas.interpretation import Interpretation

        root = Path("dataset_builder/output/_tmp_aspect_memory_pipeline")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        out = root / "out"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(root / "memory.json"),
        )
        rows = [
            BenchmarkRow(
                review_id=f"r{i}",
                group_id=f"g{i}",
                domain="restaurant",
                domain_family="dining",
                review_text="The pizza was great",
                gold_interpretations=(
                    interp(
                        aspect_raw="pizza",
                        aspect_canonical="unknown",
                        mapping_source="unmapped",
                        sentiment="positive",
                    ),
                ),
            )
            for i in range(1, 6)
        ]
        try:
            CanonicalizationStage().process(rows, cfg)
            summary_path = out / "aspect_memory_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("total_entries", summary)
            self.assertIn("top_clusters", summary)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_pipeline_exports_domain_holdout_counterfactual_and_rejected_audit(self) -> None:
        from dataset_builder.orchestrator.pipeline import run_builder_pipeline

        root = Path("dataset_builder/output/_tmp_export_layout")
        if root.exists():
            import shutil
            shutil.rmtree(root)
        out = root / "out"
        cfg = BuilderConfig(
            input_paths=(),
            output_dir=out,
            llm_provider="none",
            overwrite=True,
            aspect_memory_path=str(root / "memory.json"),
        )
        cfg.__dict__["_rejected_rows_audit"] = [
            {
                "review_id": "rejected-1",
                "domain": "electronics",
                "review_text": "Unknown thing.",
                "stage": "canonicalization",
                "reason": "empty_gold_after_canonicalization",
                "raw_candidates": [],
                "dropped_candidates": [],
                "recommended_recovery": "open_world_candidate",
            }
        ]
        rows_by_split = {
            "train": [
                BenchmarkRow(
                    review_id="r1",
                    group_id="g1",
                    domain="electronics",
                    domain_family="electronics",
                    review_text="Battery works great.",
                    gold_interpretations=[interp(aspect_raw="battery", aspect_canonical="battery_life", evidence_text="Battery", evidence_span=[0, 7])],
                )
            ],
            "val": [
                BenchmarkRow(
                    review_id="r2",
                    group_id="g2",
                    domain="restaurant",
                    domain_family="dining",
                    review_text="The food was good.",
                    gold_interpretations=[interp(aspect_raw="food", aspect_canonical="food_quality", evidence_text="food", evidence_span=[4, 8])],
                )
            ],
            "test": [
                BenchmarkRow(
                    review_id="r3",
                    group_id="g3",
                    domain="electronics",
                    domain_family="electronics",
                    review_text="Screen is bright.",
                    gold_interpretations=[interp(aspect_raw="screen", aspect_canonical="display", evidence_text="Screen", evidence_span=[0, 6])],
                )
            ],
        }

        with patch("dataset_builder.orchestrator.pipeline.assert_release_ready") as gate:
            gate.return_value = {"status": "PASS", "failures": [], "warnings": [], "metrics": {}}
            result = run_builder_pipeline(cfg, rows_by_split=rows_by_split)

        self.assertIn("counts", result)
        self.assertTrue((out / "train.jsonl").exists())
        self.assertTrue((out / "val.jsonl").exists())
        self.assertTrue((out / "test.jsonl").exists())
        self.assertTrue((out / "domain_holdout" / "train.jsonl").exists())
        self.assertTrue((out / "domain_holdout" / "val.jsonl").exists())
        self.assertTrue((out / "domain_holdout" / "test.jsonl").exists())
        self.assertTrue((out / "counterfactual" / "counterfactual_pairs.jsonl").exists())
        self.assertTrue((out / "counterfactual" / "originals.jsonl").exists())
        self.assertTrue((out / "counterfactual" / "counterfactuals.jsonl").exists())
        self.assertTrue((out / "rejected_rows.jsonl").exists())
        self.assertTrue((out / "grouped" / "train.jsonl").exists())
        self.assertTrue((out / "grouped" / "val.jsonl").exists())
        self.assertTrue((out / "grouped" / "test.jsonl").exists())
        self.assertGreater((out / "counterfactual" / "counterfactual_pairs.jsonl").stat().st_size, 0)
        with zipfile.ZipFile(out / "artifact.zip") as archive:
            names = set(archive.namelist())
        self.assertIn("grouped/train.jsonl", names)
        self.assertIn("domain_holdout/train.jsonl", names)
        self.assertIn("counterfactual/counterfactual_pairs.jsonl", names)
        self.assertIn("rejected_rows.jsonl", names)

    def test_benchmark_stage_records_recommended_recovery_for_empty_gold_rows(self) -> None:
        from dataset_builder.orchestrator.stages import BenchmarkStage

        cfg = BuilderConfig()
        row = BenchmarkRow(
            review_id="r-empty",
            group_id="g-empty",
            domain="restaurant",
            domain_family="dining",
            review_text="Great evening.",
            gold_interpretations=(),
            candidate_trace={
                "after_extraction": ["great evening"],
                "after_fusion": ["great evening"],
                "after_canonicalization": ["great evening"],
                "after_pruning": [],
            },
        )

        rows = BenchmarkStage().process([row], cfg)
        self.assertEqual(rows, [])
        audit = cfg.__dict__["_rejected_rows_audit"]
        self.assertEqual(audit[0]["reason"], "empty_gold_after_canonicalization")
        self.assertIn("recommended_recovery", audit[0])

    def test_benchmark_stage_exports_controlled_vague_rows_as_abstain_examples(self) -> None:
        from dataset_builder.orchestrator.stages import BenchmarkStage

        cfg = BuilderConfig()
        row = BenchmarkRow(
            review_id="r-vague",
            group_id="g-vague",
            domain="shopping",
            domain_family="shopping",
            review_text="Something about the app was frustrating.",
            gold_interpretations=(),
            provenance={"metadata": {"fixture_type": "controlled_abstain", "fixture_priority": "must_include"}},
            candidate_trace={
                "after_extraction": [],
                "after_fusion": [],
                "after_canonicalization": [],
                "after_pruning": [],
            },
        )

        rows = BenchmarkStage().process([row], cfg)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].abstain_acceptable)
        self.assertIn("vague_review", rows[0].abstain_reason_gold)

    def test_benchmark_stage_forces_controlled_abstain_fixtures_to_remain_abstain(self) -> None:
        from dataset_builder.orchestrator.stages import BenchmarkStage

        cfg = BuilderConfig()
        row = BenchmarkRow(
            review_id="r-vague-gold",
            group_id="g-vague-gold",
            domain="shopping",
            domain_family="shopping",
            review_text="Something about the app was frustrating.",
            gold_interpretations=(
                interp(
                    aspect_raw="purchase",
                    aspect_canonical="purchase",
                    mapping_source="open_world_candidate",
                    mapping_scope="open_world_candidate",
                    canonical_confidence=0.15,
                ),
            ),
            provenance={"metadata": {"fixture_type": "controlled_abstain", "fixture_priority": "must_include"}},
            candidate_trace={
                "after_extraction": [],
                "after_fusion": [],
                "after_canonicalization": [],
                "after_pruning": [],
            },
        )

        rows = BenchmarkStage().process([row], cfg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].gold_interpretations, ())
        self.assertTrue(rows[0].abstain_acceptable)
        self.assertIn("vague_review", rows[0].abstain_reason_gold)

    def test_canonicalization_stage_feeds_open_world_behavior_rows_into_aspect_memory(self) -> None:
        from dataset_builder.orchestrator.stages import CanonicalizationStage

        root = Path("dataset_builder/output/_tmp_memory_stage")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            cfg = BuilderConfig(
                output_dir=root,
                aspect_memory_path=str(root / "aspect_memory_candidates.json"),
            )
            row = BenchmarkRow(
                review_id="r-memory-1",
                group_id="g-memory-1",
                domain="telecom",
                domain_family="telecom",
                review_text="My calls kept dropping every few minutes.",
                gold_interpretations=(
                    interp(
                        aspect_raw="calls kept dropping",
                        aspect_canonical="call_reliability",
                        latent_family="call_reliability",
                        label_type="implicit",
                        source="behavior_pattern_matcher",
                        source_type="implicit_json",
                        mapping_source="open_world_candidate",
                        mapping_scope="open_world_candidate",
                        mapping_layers=("open_world_candidate",),
                        canonical_confidence=0.85,
                        evidence_text="calls kept dropping every few minutes",
                        evidence_span=[3, 40],
                    ),
                ),
            )

            CanonicalizationStage().process([row], cfg)
            memory = AspectMemory(root / "aspect_memory_candidates.json")

            self.assertGreaterEqual(len(memory.entries), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_aspect_memory_clusters_similar_behavior_rows_under_same_discovery_label(self) -> None:
        root = Path("dataset_builder/output/_tmp_memory_cluster")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            memory = AspectMemory(root / "aspect_memory_candidates.json")
            memory.add_evidence(
                aspect_raw="call_reliability",
                review_id="r1",
                evidence_text="My calls kept dropping every few minutes.",
                domain="telecom",
            )
            memory.add_evidence(
                aspect_raw="call_reliability",
                review_id="r2",
                evidence_text="My calls kept dropping during short calls.",
                domain="telecom",
            )
            memory.add_evidence(
                aspect_raw="call_reliability",
                review_id="r3",
                evidence_text="Calls kept dropping even with full signal.",
                domain="telecom",
            )

            self.assertEqual(len(memory.entries), 1)
            entry = next(iter(memory.entries.values()))
            self.assertGreaterEqual(entry.support_count, 3)
            self.assertGreaterEqual(len(entry.trigger_patterns), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_noisy_explicit_fragments_are_rejected(self) -> None:
        cases = [
            '"service',
            "M",
            "thing",
            "what am I supposed to do",
            "provide me the service",
        ]
        for label in cases:
            self.assertTrue(is_noisy_label(label), msg=label)


class VerifyArtifactTests(unittest.TestCase):
    def test_profile_thresholds_scale_for_stability_and_development(self) -> None:
        from dataset_builder.benchmark.verification_policy import profile_thresholds

        dev = profile_thresholds("development", 200)
        stable = profile_thresholds("stability", 400)
        strict_diag = profile_thresholds("diagnostic_strict", 200)
        journal = profile_thresholds("journal", 1000)

        self.assertEqual(dev["min_counterfactual_validated"], 30)
        self.assertEqual(dev["min_anchor_modifier_count"], 20)
        self.assertEqual(dev["review_queue_min"], 1)
        self.assertEqual(dev["min_aspect_swap_count"], 3)
        self.assertEqual(stable["min_counterfactual_validated"], 60)
        self.assertEqual(stable["min_anchor_modifier_count"], 25)
        self.assertEqual(stable["review_queue_min"], 3)
        self.assertEqual(stable["min_aspect_swap_count"], 10)
        self.assertEqual(strict_diag["min_aspect_swap_count"], 3)
        self.assertEqual(journal["min_aspect_swap_count"], 25)

    def test_counterfactual_quality_report_is_required_and_must_include_type_counts(self) -> None:
        from dataset_builder.scripts.verify_artifact import verify

        root = Path("dataset_builder/output/_tmp_verify_artifact")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest = {
                "run_id": "r1",
                "run_command": "python",
                "config_hash": "c1",
                "code_hash": "h1",
                "artifact_created_at": "2026-05-06T00:00:00Z",
                "artifact_version": "v1",
                "sample_size_requested": 400,
                "sample_size_loaded": 400,
                "release_status": "passed",
                "gate_status": "PASS",
            }
            metrics = {
                "quality": {
                    "total_exported": 360,
                    "evidence": {"full_review_evidence_rate": 0.05, "matched_term_in_evidence_rate": 0.98},
                    "canonicalization": {"anchor_modifier_count": 40, "row_metadata_unknown_count": 0, "mapping_scope_unknown_count": 0},
                    "novelty_distribution": {"known": 310, "boundary": 20, "novel": 30},
                    "hardness_distribution": {"H1": 270, "H2": 70, "H3": 20},
                },
                "aspect_memory": {"review_queue_count": 3, "unknown_candidate_count": 0, "broad_noun_candidate_rate": 0.1},
                "domain_holdout": {"counts": {"val": 20}},
                "counterfactual_pairs": {
                    "total": 50,
                    "stats": {
                        "attempted": 80,
                        "generated": 50,
                        "exported": 50,
                        "validated": 50,
                        "rejected_unnatural": 10,
                        "rejected_no_expected_change": 20,
                        "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            for rel in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "quality_report.json",
                "rejected_rows.jsonl",
                "aspect_memory_summary.json",
                "domain_holdout/manifest.json",
                "domain_holdout/train.jsonl",
                "domain_holdout/val.jsonl",
                "domain_holdout/test.jsonl",
                "counterfactual/counterfactual_pairs.jsonl",
                "counterfactual/originals.jsonl",
                "counterfactual/counterfactuals.jsonl",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / "counterfactual" / "counterfactual_quality_report.json").write_text(
                json.dumps({
                    "attempted": 80,
                    "generated": 50,
                    "exported": 50,
                    "validated": 50,
                    "rejected_unnatural": 10,
                    "rejected_no_expected_change": 20,
                    "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                }),
                encoding="utf-8",
            )
            result = verify(str(root), profile="stability", expected_rows=400)
            self.assertEqual(result, 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_verify_artifact_requires_aspect_swaps_for_stability_profile(self) -> None:
        from dataset_builder.scripts.verify_artifact import verify

        root = Path("dataset_builder/output/_tmp_verify_artifact_aspect_swaps")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest = {
                "run_id": "r1",
                "run_command": "python",
                "config_hash": "c1",
                "code_hash": "h1",
                "artifact_created_at": "2026-05-06T00:00:00Z",
                "artifact_version": "v1",
                "sample_size_requested": 400,
                "sample_size_loaded": 400,
                "release_status": "passed",
                "gate_status": "PASS",
            }
            metrics = {
                "quality": {
                    "total_exported": 360,
                    "evidence": {"full_review_evidence_rate": 0.05, "matched_term_in_evidence_rate": 0.98},
                    "canonicalization": {"anchor_modifier_count": 40, "row_metadata_unknown_count": 0, "mapping_scope_unknown_count": 0},
                    "novelty_distribution": {"known": 310, "boundary": 20, "novel": 30},
                    "hardness_distribution": {"H1": 270, "H2": 70, "H3": 20},
                },
                "aspect_memory": {"review_queue_count": 3, "unknown_candidate_count": 0, "broad_noun_candidate_rate": 0.1},
                "domain_holdout": {"counts": {"val": 20}},
                "counterfactual_pairs": {
                    "total": 50,
                    "stats": {
                        "attempted": 80,
                        "generated": 50,
                        "exported": 50,
                        "validated": 50,
                        "rejected_unnatural": 10,
                        "rejected_no_expected_change": 20,
                        "type_counts": {"aspect_swap": 1, "sentiment_flip": 49},
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            for rel in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "rejected_rows.jsonl",
                "aspect_memory_summary.json",
                "domain_holdout/manifest.json",
                "domain_holdout/train.jsonl",
                "domain_holdout/val.jsonl",
                "domain_holdout/test.jsonl",
                "counterfactual/counterfactual_pairs.jsonl",
                "counterfactual/originals.jsonl",
                "counterfactual/counterfactuals.jsonl",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / "counterfactual").mkdir(parents=True, exist_ok=True)
            (root / "counterfactual" / "counterfactual_quality_report.json").write_text(
                json.dumps({
                    "attempted": 80,
                    "generated": 50,
                    "exported": 50,
                    "validated": 50,
                    "rejected_unnatural": 10,
                    "rejected_no_expected_change": 20,
                    "type_counts": {"aspect_swap": 1, "sentiment_flip": 49},
                }),
                encoding="utf-8",
            )
            out = verify(str(root), profile="stability")
            self.assertEqual(out, 1)
            report = json.loads((root / "artifact_verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["artifact_status"], "fail")
            self.assertTrue(any("aspect_swap_count" in failure for failure in report["failed_checks"]))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_verify_artifact_reports_strict_novel_and_boundary_rates_separately(self) -> None:
        from dataset_builder.scripts.verify_artifact import verify

        root = Path("dataset_builder/output/_tmp_verify_artifact_novel")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest = {
                "run_id": "r1",
                "run_command": "python",
                "config_hash": "c1",
                "code_hash": "h1",
                "artifact_created_at": "2026-05-06T00:00:00Z",
                "artifact_version": "v1",
                "sample_size_requested": 400,
                "sample_size_loaded": 400,
                "release_status": "passed",
                "gate_status": "PASS",
            }
            metrics = {
                "quality": {
                    "total_exported": 360,
                    "evidence": {"full_review_evidence_rate": 0.05, "matched_term_in_evidence_rate": 0.98},
                    "canonicalization": {"anchor_modifier_count": 40, "row_metadata_unknown_count": 0, "mapping_scope_unknown_count": 0},
                    "novelty_distribution": {"known": 300, "boundary": 30, "novel": 30},
                    "hardness_distribution": {"H1": 280, "H2": 50, "H3": 30},
                },
                "aspect_memory": {"review_queue_count": 3, "unknown_candidate_count": 0, "broad_noun_candidate_rate": 0.1},
                "domain_holdout": {"counts": {"val": 20}},
                "counterfactual_pairs": {
                    "total": 50,
                    "stats": {
                        "attempted": 80,
                        "generated": 50,
                        "exported": 50,
                        "validated": 50,
                        "rejected_unnatural": 10,
                        "rejected_no_expected_change": 20,
                        "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            for rel in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "rejected_rows.jsonl",
                "aspect_memory_summary.json",
                "domain_holdout/manifest.json",
                "domain_holdout/train.jsonl",
                "domain_holdout/val.jsonl",
                "domain_holdout/test.jsonl",
                "counterfactual/counterfactual_pairs.jsonl",
                "counterfactual/originals.jsonl",
                "counterfactual/counterfactuals.jsonl",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / "counterfactual").mkdir(parents=True, exist_ok=True)
            (root / "counterfactual" / "counterfactual_quality_report.json").write_text(
                json.dumps({
                    "attempted": 80,
                    "generated": 50,
                    "exported": 50,
                    "validated": 50,
                    "rejected_unnatural": 10,
                    "rejected_no_expected_change": 20,
                    "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                }),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "artifact.zip", "w") as archive:
                archive.write(root / "manifest.json", arcname="manifest.json")
                archive.write(root / "metrics_summary.json", arcname="metrics_summary.json")
                archive.write(root / "quality_report.json", arcname="quality_report.json")
            out = verify(str(root), profile="stability")
            self.assertEqual(out, 0)
            report = json.loads((root / "artifact_verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["artifact_status"], "pass")
            self.assertEqual(report["metrics"]["strict_novel_rate"], 30 / 360)
            self.assertEqual(report["metrics"]["boundary_rate"], 30 / 360)
            with zipfile.ZipFile(root / "artifact.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("artifact_verification.json", names)
            self.assertIn("source_artifact_consistency.json", names)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_verify_artifact_keeps_source_consistency_separate_from_quality_status(self) -> None:
        from dataset_builder.scripts.verify_artifact import verify

        root = Path("dataset_builder/output/_tmp_verify_artifact_consistency")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest = {
                "run_id": "r1",
                "run_command": "python",
                "config_hash": "c1",
                "code_hash": "h1",
                "artifact_created_at": "2026-05-06T00:00:00Z",
                "artifact_version": "v1",
                "sample_size_requested": 400,
                "sample_size_loaded": 400,
                "release_status": "passed",
                "gate_status": "PASS",
            }
            metrics = {
                "quality": {
                    "total_exported": 360,
                    "evidence": {"full_review_evidence_rate": 0.25, "matched_term_in_evidence_rate": 0.98},
                    "canonicalization": {"anchor_modifier_count": 40, "row_metadata_unknown_count": 0, "mapping_scope_unknown_count": 0},
                    "novelty_distribution": {"known": 320, "boundary": 20, "novel": 20},
                    "hardness_distribution": {"H1": 280, "H2": 50, "H3": 30},
                },
                "aspect_memory": {"review_queue_count": 3, "unknown_candidate_count": 0, "broad_noun_candidate_rate": 0.1},
                "domain_holdout": {"counts": {"val": 20}},
                "counterfactual_pairs": {
                    "total": 50,
                    "stats": {
                        "attempted": 80,
                        "generated": 50,
                        "exported": 50,
                        "validated": 50,
                        "rejected_unnatural": 10,
                        "rejected_no_expected_change": 20,
                        "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            for rel in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "rejected_rows.jsonl",
                "aspect_memory_summary.json",
                "domain_holdout/manifest.json",
                "domain_holdout/train.jsonl",
                "domain_holdout/val.jsonl",
                "domain_holdout/test.jsonl",
                "counterfactual/counterfactual_pairs.jsonl",
                "counterfactual/originals.jsonl",
                "counterfactual/counterfactuals.jsonl",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / "counterfactual").mkdir(parents=True, exist_ok=True)
            (root / "counterfactual" / "counterfactual_quality_report.json").write_text(
                json.dumps({
                    "attempted": 80,
                    "generated": 50,
                    "exported": 50,
                    "validated": 50,
                    "rejected_unnatural": 10,
                    "rejected_no_expected_change": 20,
                    "type_counts": {"aspect_swap": 40, "sentiment_flip": 10},
                }),
                encoding="utf-8",
            )
            out = verify(str(root), profile="stability")
            self.assertEqual(out, 1)
            report = json.loads((root / "artifact_verification.json").read_text(encoding="utf-8"))
            self.assertEqual(report["artifact_status"], "fail")
            self.assertEqual(report["quality_status"], "fail")
            self.assertTrue(report["source_artifact_consistency"]["artifact_matches_source"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_verify_artifact_supports_strict_followup_flags(self) -> None:
        from dataset_builder.scripts.verify_artifact import verify

        root = Path("dataset_builder/output/_tmp_verify_artifact_flags")
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            manifest = {
                "run_id": "r1",
                "run_command": "python",
                "config_hash": "c1",
                "code_hash": "h1",
                "artifact_created_at": "2026-05-06T00:00:00Z",
                "artifact_version": "v1",
                "sample_size_requested": 400,
                "sample_size_loaded": 400,
                "release_status": "passed",
                "gate_status": "PASS",
            }
            metrics = {
                "quality": {
                    "total_exported": 360,
                    "rejected_rows": 2,
                    "evidence": {"full_review_evidence_rate": 0.05, "matched_term_in_evidence_rate": 0.98},
                    "canonicalization": {"anchor_modifier_count": 40, "row_metadata_unknown_count": 0, "mapping_scope_unknown_count": 0},
                    "novelty_distribution": {"known": 300, "boundary": 30, "novel": 30},
                    "hardness_distribution": {"H1": 280, "H2": 50, "H3": 30},
                },
                "aspect_memory": {
                    "review_queue_count": 3,
                    "unknown_candidate_count": 0,
                    "broad_noun_candidate_rate": 0.1,
                    "bootstrap_entry_count": 0,
                    "organic_entry_count": 5,
                    "bootstrap_review_queue_count": 0,
                    "organic_review_queue_count": 3,
                },
                "domain_holdout": {"counts": {"val": 20}},
                "counterfactual_pairs": {
                    "total": 50,
                    "stats": {
                        "attempted": 80,
                        "generated": 50,
                        "exported": 50,
                        "validated": 50,
                        "rejected_unnatural": 10,
                        "rejected_no_expected_change": 20,
                        "type_counts": {"aspect_swap": 10, "sentiment_flip": 40},
                        "source_counts": {"natural_match": 45, "synthetic_seed": 5, "template_seed": 0},
                        "natural_aspect_swap_count": 6,
                        "synthetic_aspect_swap_count": 4,
                    },
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "metrics_summary.json").write_text(json.dumps(metrics), encoding="utf-8")
            (root / "quality_report.json").write_text(json.dumps(metrics["quality"]), encoding="utf-8")
            for rel in (
                "train.jsonl",
                "val.jsonl",
                "test.jsonl",
                "aspect_memory_summary.json",
                "domain_holdout/manifest.json",
                "domain_holdout/train.jsonl",
                "domain_holdout/val.jsonl",
                "domain_holdout/test.jsonl",
                "counterfactual/counterfactual_pairs.jsonl",
                "counterfactual/originals.jsonl",
                "counterfactual/counterfactuals.jsonl",
            ):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (root / "rejected_rows.jsonl").write_text(
                json.dumps({"review_id": "r1", "candidate_trace": {}, "recommended_recovery": "abstain"}) + "\n" +
                json.dumps({"review_id": "r2", "candidate_trace": {}, "recommended_recovery": "reject_noise"}) + "\n",
                encoding="utf-8",
            )
            (root / "counterfactual" / "counterfactual_quality_report.json").write_text(
                json.dumps(metrics["counterfactual_pairs"]["stats"]),
                encoding="utf-8",
            )
            out = verify(
                str(root),
                profile="stability",
                expected_rows=400,
                require_organic_memory=True,
                require_rejected_row_audit=True,
                require_counterfactual_source_breakdown=True,
            )
            self.assertEqual(out, 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
