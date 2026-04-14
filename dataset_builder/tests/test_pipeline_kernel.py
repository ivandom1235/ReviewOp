from __future__ import annotations

import sys
import unittest
import shutil
from pathlib import Path
from types import SimpleNamespace

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from exporters import write_pipeline_outputs
from pipeline_state import build_pipeline_state
from report_blockers import finalize_report
from report_context import build_report_context
from report_payload import assemble_pipeline_report


class PipelineKernelTests(unittest.TestCase):
    def test_build_report_context_is_explicit_and_contains_core_inputs(self) -> None:
        cfg = SimpleNamespace(output_version="v1")
        context = build_report_context(
            cfg=cfg,
            generated_at="2024-01-01T00:00:00Z",
            run_profile="research",
            artifact_mode="research_release",
            config={"output_version": "v1"},
            text_column="body_text",
            frame=[{"id": "row-1"}],
            prepared=[{"id": "row-1"}],
            sample_frame=[{"id": "row-1"}],
            train_built=[{"id": "train-1"}],
            val_built=[],
            test_built=[],
            finalized_rows=[{"id": "row-1"}],
            candidate_aspects=["performance"],
            candidate_aspects_by_language={"en": ["performance"]},
            candidate_aspects_by_domain_train={"electronics": ["performance"]},
            chunk_preview=[{"id": "row-1"}],
            prepared_language_distribution={"en": 1},
            schema=SimpleNamespace(schema_fingerprint="schema-1"),
            train_domain_conditioning_mode="adaptive_soft",
            eval_domain_conditioning_mode="adaptive_soft",
            research={"benchmark": "demo"},
            diagnostics={"rows": 1},
            pipeline_state={
                "train": {"export_rows": [], "stage_counts": {}, "sentiment_constraints": {}, "topup_stats": {}, "general_policy_stats": {}, "review_filter_stats": {}, "reinference_stats": {}, "quarantine_stats": {}},
                "benchmark": {"metadata": {}, "gold_metrics": {}, "structural_audits": {}, "novelty": {}, "rows_by_split": {}, "review_queue_rows": []},
                "evaluation": {"robust_training_eval": {}, "promotion_guard": {}, "gold_metrics": {}, "domain_generalization": {}, "unseen_metrics": {}},
                "governance": {"sentiment_quality": {}},
            },
            train_review_filter_stats={"train_review_rows_before_filter": 1, "train_review_rows_after_filter": 1, "train_review_filter_applied": False},
            train_quarantine_recoverable_rows=[],
            train_quarantine_stats={"recoverable_rows": 0},
            train_review_dropped_soft_rows=[],
            train_review_dropped_hard_rows=[],
            train_salvage_stats={"applied": False},
            train_leakage_filter_stats_before_salvage={"removed_rows": 0},
            train_leakage_filter_stats_after_salvage={"removed_rows": 0},
            train_leakage_filter_stats_after_topup={"removed_rows": 0},
            train_leakage_filter_stats_after_targeting={"removed_rows": 0},
            train_sentiment_before_balance={"neutral": 1},
            train_sentiment_after_balance={"neutral": 1},
            train_general_dominance_rate=0.0,
            train_domain_leakage_metrics={"train_domain_leakage_row_rate": 0.0},
            eval_domain_leakage_metrics={"eval_domain_leakage_row_rate": 0.0},
            train_negative_ratio=0.0,
            train_positive_ratio=0.0,
            train_neutral_ratio=1.0,
            train_target_blocking_failure=False,
            sampled_run_blocked_or_debug=False,
            quality_analysis_summary={"recoverable_count": 0},
            explicit_metrics={"rows": 1},
            counts_match=True,
            run_registry={"domains": {}},
            promoted_registry={"domains": {}},
            run_registry_version="run-v1",
            promoted_registry_version="promoted-v1",
            benchmark_spec=SimpleNamespace(key="demo-benchmark", family="demo-family"),
            model_spec=SimpleNamespace(key="demo-model", kind="demo-kind"),
            benchmark_rows_by_split={"train": [], "val": [], "test": []},
            benchmark_metadata={"rows": 0},
            core_benchmark_domains=["electronics"],
            synthetic_audit={"accepted": 0, "rejected": 0},
            strict_train_export_rows=[{"id": "train-1"}],
            strict_val_export_rows=[],
            strict_test_export_rows=[],
            strict_review_queue_rows=[],
            strict_challenge_rows=[],
            strict_floor_stats={"applied": False},
            train_export_floor_rows=[],
            grounding={"grounded_prediction_rate": 1.0, "ungrounded_non_general_count": 0},
            domain_prior_boost_count=0,
            domain_prior_penalty_count=0,
        )

        self.assertEqual(context.cfg.output_version, "v1")
        self.assertEqual(context.generated_at, "2024-01-01T00:00:00Z")
        self.assertEqual(context.text_column, "body_text")
        self.assertEqual(context.candidate_aspects_by_domain_train["electronics"], ["performance"])
        self.assertEqual(context.pipeline_state["train"]["stage_counts"], {})
        self.assertTrue(context.counts_match)
        self.assertEqual(context.core_benchmark_domains, ["electronics"])

    def test_write_pipeline_outputs_persists_core_artifacts(self) -> None:
        output_dir = Path.cwd() / "dataset_builder" / "_tmp_pipeline_outputs"
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = SimpleNamespace(
                output_dir=output_dir,
                benchmark_dir=output_dir / "benchmark" / "ambiguity_grounded",
                reports_dir=output_dir / "reports",
                emit_review_set=False,
                review_set_size=0,
                random_seed=42,
            )
            report = {
                "validation": {
                    "benchmark_artifact_counts_match": True,
                    "sampled_run_blocked_or_debug": False,
                    "train_target_blocking_failure": False,
                    "train_domain_leakage_ok": True,
                    "train_general_excluded": True,
                    "no_generic_aspects": True,
                    "no_rejected_aspects": True,
                    "train_positive_ratio_within_max": True,
                    "train_neutral_ratio_within_max": True,
                    "strict_explicit_contamination_ok": True,
                    "strict_boundary_fp_ok": True,
                    "strict_h2_h3_ok": True,
                    "strict_multi_aspect_ok": True,
                    "strict_challenge_ok": True,
                    "grouped_split_leakage_ok": True,
                    "benchmark_val_non_empty": True,
                    "benchmark_grounded_evidence_ok": True,
                    "benchmark_duplicate_rate_ok": True,
                    "benchmark_thermal_share_ok": True,
                    "benchmark_domain_coverage_ok": True,
                    "benchmark_family_floor_ok": True,
                    "benchmark_implicit_purity_ok": True,
                    "benchmark_ontology_compatibility_ok": True,
                    "sentiment_mismatch_rate_ok": True,
                    "promotion_guard_ok": True,
                },
                "row_counts": {"train_export": 1},
                "train_viability_guard_triggered": False,
                "generated_at": "2024-01-01T00:00:00Z",
                "domain_prior_boost_count": 0,
                "domain_prior_penalty_count": 0,
            }

            write_pipeline_outputs(
                cfg=cfg,
                report=report,
                benchmark_rows_by_split={"train": [{"id": "bench-train"}], "val": [], "test": []},
                benchmark_metadata={"split_counts": {"train": 1, "val": 0, "test": 0}, "rows": 1},
                benchmark_protocol_views={"random": {"train": [{"id": "bench-train"}], "val": [], "test": []}},
                benchmark_review_queue_rows=[{"id": "review-1"}],
                run_registry={"domains": {"electronics": 1}},
                promoted_registry={"domains": {"electronics": 1}},
                quality_analysis_artifact={"train_rows": 1},
                synthetic_accepted=[],
                synthetic_rejected=[],
                synthetic_audit={"accepted": 0, "rejected": 0},
                benchmark_v2_novelty={"novelty": []},
                research_manifest={"manifest": True},
                previous_accepted_path=output_dir / "state" / "accepted_training_metrics.json",
                robust_training_eval={"groupdro": {"worst_domain_f1": 0.8}},
                promotion_guard={"blocked": False},
            )

            self.assertTrue((output_dir / "reports" / "build_report.json").exists())
            self.assertTrue((output_dir / "reports" / "data_quality_report.json").exists())
            self.assertTrue((output_dir / "benchmark" / "ambiguity_grounded" / "metadata.json").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_build_pipeline_state_groups_pipeline_outputs(self) -> None:
        state = build_pipeline_state(
            train={
                "export_rows": [{"id": "train-1"}],
                "stage_counts": {"start": 1, "after_review_filter": 1},
            },
            benchmark={
                "rows_by_split": {"train": [], "val": [{"id": "bench-1"}], "test": []},
                "metadata": {"rows": 1},
                "review_queue_rows": [{"id": "review-1"}],
            },
            evaluation={
                "gold_metrics": {"has_gold_labels": True},
                "benchmark_gold_metrics": {"has_gold_interpretations": True},
                "robust_training_eval": {"groupdro": {"worst_domain_f1": 0.8}},
                "promotion_guard": {"blocked": False},
            },
            governance={
                "sentiment_quality": {"sentiment_mismatch_rate": 0.1},
                "train_target_stats": {"size_within_target_range": True},
            },
        )

        self.assertEqual(state["train"]["stage_counts"]["after_review_filter"], 1)
        self.assertEqual(state["benchmark"]["metadata"]["rows"], 1)
        self.assertEqual(state["evaluation"]["gold_metrics"]["has_gold_labels"], True)
        self.assertEqual(state["governance"]["sentiment_quality"]["sentiment_mismatch_rate"], 0.1)

    def test_assemble_pipeline_report_builds_payload_without_blockers(self) -> None:
        cfg = SimpleNamespace(
            output_version="v1",
            implicit_mode="zeroshot",
            multilingual_mode="shared_vocab",
            use_coref=False,
            train_max_positive_ratio=0.6,
            train_neutral_max_ratio=0.8,
            unseen_non_general_coverage_min=0.0,
            unseen_implicit_not_ready_rate_max=1.0,
            unseen_domain_leakage_row_rate_max=1.0,
            strict_explicit_in_implicit_rate_max=1.0,
            strict_boundary_fp_max=10,
            strict_h2_h3_ratio_min=0.0,
            strict_multi_aspect_ratio_min=0.0,
            strict_challenge_macro_f1_min=0.0,
        )
        context = build_report_context(
            cfg=cfg,
            generated_at="2024-01-01T00:00:00Z",
            run_profile="research",
            artifact_mode="research_release",
            config={"output_version": "v1"},
            text_column="body_text",
            frame=[{"id": "row-1"}],
            prepared=[{"id": "row-1"}],
            sample_frame=[{"id": "row-1"}],
            train_built=[{"id": "train-1"}],
            val_built=[],
            test_built=[],
            finalized_rows=[{"id": "row-1"}],
            candidate_aspects=["performance"],
            candidate_aspects_by_language={"en": ["performance"]},
            candidate_aspects_by_domain_train={"electronics": ["performance"]},
            chunk_preview=[{"id": "row-1"}],
            prepared_language_distribution={"en": 1},
            schema=SimpleNamespace(schema_fingerprint="schema-1"),
            train_domain_conditioning_mode="adaptive_soft",
            eval_domain_conditioning_mode="adaptive_soft",
            research={"benchmark": "demo"},
            diagnostics={"rows": 1},
            pipeline_state={
                "train": {
                    "export_rows": [{"id": "train-1"}],
                    "stage_counts": {"start": 1},
                    "sentiment_constraints": {"viability_guard_triggered": False},
                    "topup_stats": {},
                    "target_stats": {"size_within_target_range": True},
                    "general_policy_stats": {"train_general_rows_before_policy": 0, "train_general_rows_after_policy": 0, "train_general_policy_applied": False},
                    "review_filter_stats": {},
                    "reinference_stats": {},
                    "quarantine_stats": {},
                },
                "benchmark": {"metadata": {}, "gold_metrics": {}, "structural_audits": {}, "novelty": {}, "rows_by_split": {}, "review_queue_rows": []},
                "evaluation": {"robust_training_eval": {}, "promotion_guard": {}, "gold_metrics": {}, "domain_generalization": {}, "unseen_metrics": {}},
                "governance": {"sentiment_quality": {}},
            },
            train_review_filter_stats={"train_review_rows_before_filter": 1, "train_review_rows_after_filter": 1, "train_review_filter_applied": False},
            train_quarantine_recoverable_rows=[],
            train_quarantine_stats={"recoverable_rows": 0},
            train_review_dropped_soft_rows=[],
            train_review_dropped_hard_rows=[],
            train_salvage_stats={"applied": False},
            train_leakage_filter_stats_before_salvage={"removed_rows": 0},
            train_leakage_filter_stats_after_salvage={"removed_rows": 0},
            train_leakage_filter_stats_after_topup={"removed_rows": 0},
            train_leakage_filter_stats_after_targeting={"removed_rows": 0},
            train_sentiment_before_balance={"neutral": 1},
            train_sentiment_after_balance={"neutral": 1},
            train_general_dominance_rate=0.0,
            train_domain_leakage_metrics={"train_domain_leakage_row_rate": 0.0},
            eval_domain_leakage_metrics={"eval_domain_leakage_row_rate": 0.0},
            train_negative_ratio=0.0,
            train_positive_ratio=0.0,
            train_neutral_ratio=1.0,
            train_target_blocking_failure=False,
            sampled_run_blocked_or_debug=False,
            quality_analysis_summary={"recoverable_count": 0, "generic_implicit_aspects": 0, "rejected_implicit_aspects": 0, "explicit_in_implicit_rate": 0.0, "boundary_false_positive_count": 0, "h2_h3_ratio": 1.0, "multi_aspect_ratio": 1.0, "challenge_macro_f1": 1.0, "strict_quality": {}},
            explicit_metrics={"rows": 1},
            counts_match=True,
            run_registry={"domains": {}},
            promoted_registry={"domains": {}},
            run_registry_version="run-v1",
            promoted_registry_version="promoted-v1",
            benchmark_spec=SimpleNamespace(key="demo-benchmark", family="demo-family"),
            model_spec=SimpleNamespace(key="demo-model", kind="demo-kind"),
            benchmark_rows_by_split={"train": [], "val": [{"id": "bench-1"}], "test": []},
            benchmark_metadata={"rows": 1, "grounded_evidence_rate": 1.0, "duplicate_interpretation_rate": 0.0, "thermal_share": 0.0, "benchmark_domain_coverage_ok": True, "family_floor_policy": {"applied": True}, "implicit_purity_rate": 1.0, "ontology_compatibility_rate": 1.0, "split_counts": {"train": 0, "val": 1, "test": 0}, "grouped_split_leakage": {"overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0}}},
            core_benchmark_domains=["electronics"],
            synthetic_audit={"accepted": 0, "rejected": 0},
            strict_train_export_rows=[{"id": "train-1"}],
            strict_val_export_rows=[],
            strict_test_export_rows=[],
            strict_review_queue_rows=[],
            strict_challenge_rows=[],
            strict_floor_stats={"applied": False},
            train_export_floor_rows=[],
            grounding={"grounded_prediction_rate": 1.0, "ungrounded_non_general_count": 0},
            domain_prior_boost_count=0,
            domain_prior_penalty_count=0,
        )

        report = assemble_pipeline_report(context=context)

        self.assertEqual(report["generated_at"], "2024-01-01T00:00:00Z")
        self.assertEqual(report["text_column"], "body_text")
        self.assertEqual(report["row_counts"]["train_export"], 1)
        self.assertNotIn("blocking_reasons", report)

    def test_finalize_report_preserves_payload_and_builds_blockers(self) -> None:
        report = {
            "validation": {
                "sampled_run_blocked_or_debug": False,
                "train_target_blocking_failure": False,
                "train_domain_leakage_ok": True,
                "train_general_excluded": True,
                "no_generic_aspects": True,
                "no_rejected_aspects": True,
                "train_positive_ratio_within_max": True,
                "train_neutral_ratio_within_max": False,
                "strict_explicit_contamination_ok": True,
                "strict_boundary_fp_ok": True,
                "strict_h2_h3_ok": True,
                "strict_multi_aspect_ok": True,
                "strict_challenge_ok": True,
                "grouped_split_leakage_ok": True,
                "benchmark_val_non_empty": True,
                "benchmark_grounded_evidence_ok": True,
                "benchmark_duplicate_rate_ok": True,
                "benchmark_thermal_share_ok": True,
                "benchmark_domain_coverage_ok": False,
                "benchmark_family_floor_ok": True,
                "benchmark_implicit_purity_ok": False,
                "benchmark_ontology_compatibility_ok": True,
                "sentiment_mismatch_rate_ok": True,
                "promotion_guard_ok": False,
                "benchmark_artifact_counts_match": True,
            },
            "size_recovery_shortfall_remaining": 0,
            "row_counts": {"train_export": 31},
            "train_viability_guard_triggered": True,
        }

        finalized = finalize_report(
            report,
            run_profile="research",
            sampled_run=False,
            train_topup_recovery_mode="strict_topup",
        )

        self.assertEqual(finalized["row_counts"]["train_export"], 31)
        self.assertTrue(finalized["train_viability_guard_triggered"])
        self.assertIn("blocking_reasons", finalized)
        self.assertEqual(
            [reason["code"] for reason in finalized["blocking_reasons"]],
            [
                "TRAIN_NEUTRAL_RATIO_TOO_HIGH",
                "BENCHMARK_DOMAIN_COVERAGE",
                "BENCHMARK_IMPLICIT_PURITY",
                "WORST_DOMAIN_REGRESSION",
            ],
        )


if __name__ == "__main__":
    unittest.main()
