import asyncio
import io
import os
import shutil
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "dataset_builder" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
PROTONET_CODE_DIR = ROOT / "protonet" / "code"
if str(PROTONET_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTONET_CODE_DIR))

from implicit_pipeline import VectorAspectMatcher, build_implicit_row  # noqa: E402
from implicit_pipeline import discover_aspects  # noqa: E402
from implicit_pipeline import infer_sentiment_details  # noqa: E402
from build_dataset import _benchmark_v2_novelty_sidecar, _build_benchmark_instances, _build_quality_analysis_artifact, _enforce_benchmark_family_floor, _group_identity, _safe_absolute_span, build_parser  # noqa: E402
from build_dataset import _merge_gold_labels  # noqa: E402
from build_dataset import _select_working_rows, _split_train_review_filter, _strict_topup_recovery, _train_floor_row_passes  # noqa: E402
from build_dataset import run_pipeline  # noqa: E402
from aspect_registry import build_run_registry, update_promoted_registry  # noqa: E402
from contracts import BuilderConfig  # noqa: E402
from evaluation import benchmark_gold_eval, gold_eval  # noqa: E402
from io_utils import load_gold_annotations, load_inputs  # noqa: E402
from llm_utils import AsyncRunPodProvider  # noqa: E402
from llm_utils import resolve_processor_async_provider  # noqa: E402
from robustness_eval import evaluate_training_tracks  # noqa: E402
from run_experiment import _resolve_artifact_mode  # noqa: E402
from splitter import grouped_leakage_report, grouped_split  # noqa: E402
from synthetic_generation import generate_synthetic_multidomain  # noqa: E402
from utils import write_jsonl  # noqa: E402
if str(CODE_DIR) in sys.path:
    sys.path.remove(str(CODE_DIR))
from evaluator import _project_prediction_rows  # noqa: E402


class V6RepairTests(unittest.TestCase):
    def test_group_identity_prefers_entity_keys_over_unique_id(self) -> None:
        row = {"id": "unique-row-id", "product_id": "SKU-123", "domain": "laptop", "source_file": "x.csv"}
        self.assertEqual(_group_identity(row), "sku-123")

    def test_safe_absolute_span_repairs_invalid_offsets(self) -> None:
        review = "Battery dies before evening and gets hot."
        span = _safe_absolute_span(review, "gets hot")
        self.assertEqual(span, [32, 40])

    def test_implicit_row_emits_canonical_aspects_and_diagnostics(self) -> None:
        row = {"id": "r1", "split": "train", "review": "The battery dies before evening."}
        result = asyncio.run(
            build_implicit_row(
                row,
                text_column="review",
                candidate_aspects=["power", "thermal"],
                confidence_threshold=0.6,
                row_index=0,
                domain="laptop",
                language="en",
                implicit_mode="zeroshot",
                enable_llm_fallback=False,
            )
        )
        implicit = result["implicit"]
        self.assertIn("aspects", implicit)
        self.assertIsInstance(implicit["aspects"], list)
        self.assertIn("aspect", implicit)
        self.assertIn("review_reason", implicit)
        self.assertIn("fallback_branch", implicit)
        self.assertIn("llm_parse_errors", implicit)
        self.assertIsInstance(implicit["llm_parse_errors"], list)

    def test_spans_have_normalized_evidence_shape(self) -> None:
        row = {"id": "r2", "split": "train", "review": "The screen is bright but the fan gets hot."}
        result = asyncio.run(
            build_implicit_row(
                row,
                text_column="review",
                candidate_aspects=["display quality", "thermal"],
                confidence_threshold=0.6,
                row_index=1,
                domain="laptop",
                language="en",
                implicit_mode="zeroshot",
                enable_llm_fallback=False,
            )
        )
        spans = result["implicit"].get("spans", [])
        self.assertTrue(spans)
        for span in spans:
            self.assertIn("evidence_text", span)
            self.assertIn("evidence_span", span)
            self.assertEqual(len(span["evidence_span"]), 2)

    def test_load_inputs_infers_domain_from_filename_when_missing(self) -> None:
        temp_dir = ROOT / "dataset_builder" / "tests" / "__tmp_domain_infer__"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = temp_dir / "Laptop_train.csv"
            path.write_text("id,review,aspect,polarity,from,to\n1,Great battery,battery,positive,0,13\n", encoding="utf-8")
            frame = load_inputs(temp_dir)
            self.assertIn("domain", frame.columns)
            self.assertEqual(set(frame["domain"].dropna().unique()), {"laptop"})
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_vector_aspect_matcher_prefers_cuda_when_available(self) -> None:
        with patch("implicit_pipeline.torch.cuda.is_available", return_value=True), patch("implicit_pipeline.SentenceTransformer") as mocked_model:
            matcher = VectorAspectMatcher("mini")
            self.assertEqual(matcher.device, "cuda")
            mocked_model.assert_called_once()
            self.assertEqual(mocked_model.call_args.kwargs.get("device"), "cuda")

        with patch("implicit_pipeline.torch.cuda.is_available", return_value=False), patch("implicit_pipeline.SentenceTransformer") as mocked_model:
            matcher = VectorAspectMatcher("mini")
            self.assertEqual(matcher.device, "cpu")
            mocked_model.assert_called_once()
            self.assertEqual(mocked_model.call_args.kwargs.get("device"), "cpu")

    def test_benchmark_dedup_and_grounding(self) -> None:
        row = {
            "id": "r3",
            "split": "train",
            "source_text": "Battery dies before evening and gets very hot.",
            "domain": "laptop",
            "gold_interpretations": [
                {"aspect_label": "power", "sentiment": "negative", "evidence_text": "Battery dies before evening", "annotator_support": 1},
                {"aspect_label": "power", "sentiment": "negative", "evidence_text": "Battery dies before evening", "annotator_support": 1},
                {"aspect_label": "thermal", "sentiment": "negative", "evidence_text": "hot", "annotator_support": 1},
                {"aspect_label": "thermal", "sentiment": "negative", "evidence_text": "ice", "annotator_support": 1},
            ],
            "implicit": {"aspects": ["power", "thermal"], "aspect_confidence": {"power": 0.9, "thermal": 0.8}, "spans": []},
        }
        rows_by_split, metadata, _ = _build_benchmark_instances([row], {"r3": {"random": "train", "grouped": "train", "domain_holdout": "train"}})
        populated_split = next(split for split in ("train", "val", "test") if rows_by_split.get(split))
        out = rows_by_split[populated_split][0]
        interpretations = out["gold_interpretations"]
        # Duplicate power interpretation is collapsed and ungrounded "ice" is removed.
        self.assertEqual(len(interpretations), 2)
        power = next(item for item in interpretations if item["aspect_label"] == "power")
        self.assertEqual(int(power["annotator_support"]), 2)
        self.assertGreaterEqual(float(metadata["grounded_evidence_rate"]), 0.6)

    def test_benchmark_gold_eval_tracks_benchmark_interpretations(self) -> None:
        rows = [
            {
                "id": "bench-1",
                "domain": "laptop",
                "review_text": "Battery dies before evening.",
                "gold_interpretations": [
                    {"aspect_label": "battery", "sentiment": "negative", "evidence_text": "Battery dies"},
                    {"aspect_label": "battery", "sentiment": "negative", "evidence_text": "Battery dies"},
                ],
                "gold_labels": [],
            }
        ]
        human = gold_eval(rows)
        bench = benchmark_gold_eval(rows)
        self.assertFalse(human["has_gold_labels"])
        self.assertTrue(bench["has_gold_interpretations"])
        self.assertEqual(bench["num_rows_with_gold_interpretations"], 1)
        self.assertAlmostEqual(float(bench["average_gold_interpretations"]), 2.0)
        self.assertAlmostEqual(float(bench["grounded_evidence_rate"]), 1.0)
        self.assertGreater(float(bench["duplicate_interpretation_rate"]), 0.0)
        self.assertGreaterEqual(float(bench["implicit_purity_rate"]), 0.0)
        self.assertGreaterEqual(float(bench["ontology_compatibility_rate"]), 0.0)

    def test_hybrid_sentiment_handles_superlatives_and_abstain(self) -> None:
        positive = infer_sentiment_details("This is the best service and fantastic support.")
        neutral = infer_sentiment_details("It was okay, nothing special overall.")
        self.assertEqual(positive["label"], "positive")
        self.assertFalse(positive["abstained"])
        self.assertIn(positive["risk_bucket"], {"low", "medium"})
        self.assertEqual(neutral["label"], "neutral")
        self.assertTrue(neutral["abstained"] or neutral["risk_bucket"] in {"medium", "high"})

    def test_aspect_registry_promotes_after_recurrence(self) -> None:
        rows = [
            {"domain": "restaurant", "implicit": {"spans": [{"latent_label": "sensory quality", "aspect": "taste", "sentiment": "positive"}]}},
            {"domain": "restaurant", "implicit": {"spans": [{"latent_label": "sensory quality", "aspect": "food", "sentiment": "positive"}]}},
        ] * 15
        run_a = build_run_registry(rows=rows, run_id="a", run_ts="2026-01-01T00:00:00Z")
        promoted_a = update_promoted_registry(previous=None, run_registry=run_a)
        run_b = build_run_registry(rows=rows, run_id="b", run_ts="2026-01-02T00:00:00Z")
        promoted_b = update_promoted_registry(previous=promoted_a, run_registry=run_b)
        run_c = build_run_registry(rows=rows, run_id="c", run_ts="2026-01-03T00:00:00Z")
        promoted_c = update_promoted_registry(previous=promoted_b, run_registry=run_c)
        status = promoted_c["domains"]["restaurant"]["food_quality"]["status"]
        self.assertEqual(status, "promoted")

    def test_synthetic_generation_produces_accept_reject_audit(self) -> None:
        accepted, rejected, audit = generate_synthetic_multidomain(domains=["restaurant", "telecom"], samples_per_domain=20)
        self.assertGreater(len(accepted), 0)
        self.assertIn("acceptance_rate", audit)
        self.assertIn("rejection_reason_counts", audit)
        self.assertEqual(audit["target_total"], 40)
        self.assertEqual(audit["accepted_total"] + audit["rejected_total"], 40)

    def test_robust_training_eval_tracks_exist(self) -> None:
        rows_by_split = {
            "train": [
                {"instance_id": "t1", "domain": "restaurant", "group_id": "g1", "gold_interpretations": [{"domain_canonical_aspect": "food_quality", "sentiment": "positive"}]},
                {"instance_id": "t2", "domain": "telecom", "group_id": "g2", "gold_interpretations": [{"domain_canonical_aspect": "connectivity", "sentiment": "negative"}]},
            ],
            "val": [
                {"instance_id": "v1", "domain": "restaurant", "group_id": "g3", "gold_interpretations": [{"domain_canonical_aspect": "food_quality", "sentiment": "positive"}]},
            ],
            "test": [
                {"instance_id": "e1", "domain": "telecom", "group_id": "g4", "gold_interpretations": [{"domain_canonical_aspect": "connectivity", "sentiment": "negative"}]},
            ],
        }
        out = evaluate_training_tracks(rows_by_split)
        self.assertIn("erm", out)
        self.assertIn("groupdro", out)
        self.assertIn("worst_domain_f1", out["groupdro"])

    def test_review_queue_annotation_import_round_trips_instance_id(self) -> None:
        temp_dir = ROOT / "dataset_builder" / "tests" / "__tmp_annotation_import__"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            annotations_path = temp_dir / "gold_annotations.jsonl"
            write_jsonl(
                annotations_path,
                [
                    {
                        "instance_id": "bench-queue-1",
                        "record_id": "bench-queue-1",
                        "review_text": "Battery dies before evening.",
                        "domain": "laptop",
                        "review_status": "completed",
                        "gold_interpretations": [
                            {
                                "aspect_label": "battery",
                                "sentiment": "negative",
                                "evidence_text": "Battery dies",
                            }
                        ],
                        "abstain_acceptable": True,
                    }
                ],
            )
            annotations = load_gold_annotations(annotations_path)
            merged = _merge_gold_labels(
                [
                    {
                        "id": "bench-queue-1",
                        "domain": "laptop",
                        "source_text": "Battery dies before evening.",
                        "gold_interpretations": [],
                    }
                ],
                annotations,
            )
            self.assertEqual(len(merged), 1)
            row = merged[0]
            self.assertEqual(row["annotation_source"], "completed")
            self.assertEqual(len(row["gold_interpretations"]), 1)
            self.assertEqual(row["gold_interpretations"][0]["aspect_label"], "battery")
            self.assertEqual(row["gold_interpretations"][0]["source"], "completed")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_post_aspect_projection_uses_separate_predictions(self) -> None:
        rows = [
            {
                "pred_label": "battery__negative",
                "post_aspect_pred_label": "screen__positive",
                "post_aspect_pred_labels": ["screen__positive"],
                "confidence": 0.91,
                "post_aspect_confidence": 0.77,
                "correct": False,
                "post_aspect_correct": True,
                "flex_correct": False,
                "post_aspect_flex_correct": True,
                "multi_label_overlap": 0.0,
                "post_aspect_multi_label_overlap": 1.0,
                "abstained": False,
                "post_aspect_abstained": False,
                "low_confidence": False,
                "post_aspect_low_confidence": False,
            }
        ]
        projected = _project_prediction_rows(rows, "post_aspect")
        self.assertEqual(projected[0]["pred_label"], "screen__positive")
        self.assertTrue(projected[0]["correct"])
        self.assertTrue(projected[0]["flex_correct"])

    def test_v2_novelty_sidecar_counts_and_leakage(self) -> None:
        rows_by_split = {
            "train": [
                {
                    "instance_id": "a1",
                    "domain": "laptop",
                    "group_id": "prod-a",
                    "novel_acceptable": True,
                    "novel_cluster_id": "novel_001",
                },
                {
                    "instance_id": "a2",
                    "domain": "laptop",
                    "group_id": "prod-b",
                    "novel_acceptable": False,
                },
            ],
            "val": [
                {
                    "instance_id": "b1",
                    "domain": "tablet",
                    "group_id": "prod-c",
                    "novel_acceptable": True,
                    "novel_cluster_id": "novel_002",
                }
            ],
            "test": [
                {
                    "instance_id": "c1",
                    "domain": "tablet",
                    "group_id": "prod-d",
                    "novel_acceptable": True,
                    "novel_cluster_id": "novel_001",
                }
            ],
        }
        sidecar = _benchmark_v2_novelty_sidecar(rows_by_split)
        self.assertEqual(sidecar["known_rows"], 1)
        self.assertEqual(sidecar["novel_rows"], 3)
        self.assertEqual(sidecar["novel_cluster_count"], 2)
        self.assertEqual(sidecar["cluster_leakage"]["cross_split_cluster_count"], 1)

    def test_debug_artifact_mode_caps_benchmark_rows(self) -> None:
        rows = []
        assignments = {}
        for i in range(240):
            row_id = f"r{i}"
            rows.append(
                {
                    "id": row_id,
                    "split": "train",
                    "source_text": f"Battery dies fast {i}.",
                    "domain": "laptop",
                    "implicit": {
                        "aspects": ["battery"],
                        "spans": [
                            {
                                "latent_label": "battery",
                                "evidence_text": f"Battery dies fast {i}",
                                "start_char": 0,
                                "end_char": 18,
                            }
                        ],
                    },
                }
            )
            assignments[row_id] = {"random": "train", "grouped": "train", "domain_holdout": "train"}
        benchmark_rows_by_split, metadata, _ = _build_benchmark_instances(
            rows,
            assignments,
            artifact_mode="debug_artifacts",
            debug_row_limit=180,
            seed=42,
        )
        total_rows = sum(len(split_rows) for split_rows in benchmark_rows_by_split.values())
        self.assertLessEqual(total_rows, 180)
        self.assertEqual(metadata["artifact_mode"], "debug_artifacts")
        self.assertEqual(metadata["rows"], total_rows)
        self.assertGreaterEqual(len(benchmark_rows_by_split["val"]), 1)

    def test_sampled_selection_preserves_core_domain_coverage(self) -> None:
        rows = [
            {
                "id": "elec-1",
                "split": "train",
                "source_text": "The laptop battery lasts all day.",
                "domain": "laptop",
                "abstain_acceptable": True,
                "implicit": {
                    "aspects": ["battery"],
                    "hardness_tier": "H3",
                    "dominant_sentiment": "negative",
                    "needs_review": True,
                    "review_reason": "weak_support",
                    "spans": [{"support_type": "exact"}],
                },
            },
            {
                "id": "rest-1",
                "split": "train",
                "source_text": "The pasta was excellent.",
                "domain": "restaurant",
                "implicit": {
                    "aspects": ["food"],
                    "hardness_tier": "H2",
                    "dominant_sentiment": "positive",
                    "needs_review": True,
                    "review_reason": "low_confidence",
                    "spans": [{"support_type": "exact"}],
                },
            },
            {
                "id": "tel-1",
                "split": "train",
                "source_text": "The signal drops indoors.",
                "domain": "telecom",
                "implicit": {
                    "aspects": ["signal"],
                    "hardness_tier": "H2",
                    "dominant_sentiment": "negative",
                    "needs_review": True,
                    "review_reason": "weak_support",
                    "spans": [{"support_type": "exact"}],
                },
            },
        ]
        for i in range(24):
            rows.append(
                {
                    "id": f"rest-fill-{i}",
                    "split": "train",
                    "source_text": f"The service was fine {i}.",
                    "domain": "restaurant",
                    "implicit": {
                        "aspects": ["service"],
                        "hardness_tier": "H0",
                        "dominant_sentiment": "neutral",
                        "spans": [{"support_type": "exact"}],
                    },
                }
            )

        cfg = BuilderConfig(sample_size=3, chunk_size=None, chunk_offset=0, random_seed=7)
        selected = _select_working_rows(rows, cfg)
        self.assertEqual(len(selected), 3)
        self.assertEqual({row["domain"] for row in selected}, {"laptop", "restaurant", "telecom"})
        self.assertTrue(any(row.get("abstain_acceptable") for row in selected))

    def test_benchmark_metadata_records_core_domain_coverage_and_hardness(self) -> None:
        rows = [
            {
                "id": "bench-elec",
                "split": "train",
                "source_text": "Battery dies before evening.",
                "domain": "laptop",
                "abstain_acceptable": True,
                "implicit": {
                    "aspects": ["battery"],
                    "hardness_tier": "H3",
                    "dominant_sentiment": "negative",
                    "spans": [{"latent_label": "battery", "support_type": "exact", "evidence_text": "Battery dies before evening"}],
                },
            },
            {
                "id": "bench-rest",
                "split": "train",
                "source_text": "The staff were friendly.",
                "domain": "restaurant",
                "implicit": {
                    "aspects": ["service"],
                    "hardness_tier": "H2",
                    "dominant_sentiment": "positive",
                    "spans": [{"latent_label": "service", "support_type": "exact", "evidence_text": "staff were friendly"}],
                },
            },
            {
                "id": "bench-tel",
                "split": "train",
                "source_text": "Signal drops indoors.",
                "domain": "telecom",
                "implicit": {
                    "aspects": ["signal"],
                    "hardness_tier": "H2",
                    "dominant_sentiment": "negative",
                    "spans": [{"latent_label": "signal", "support_type": "exact", "evidence_text": "Signal drops indoors"}],
                },
            },
        ]
        assignments = {
            "bench-elec": {"random": "train", "grouped": "train", "domain_holdout": "train"},
            "bench-rest": {"random": "val", "grouped": "val", "domain_holdout": "val"},
            "bench-tel": {"random": "test", "grouped": "test", "domain_holdout": "test"},
        }
        benchmark_rows_by_split, metadata, _ = _build_benchmark_instances(rows, assignments)
        flat_rows = [row for split_rows in benchmark_rows_by_split.values() for row in split_rows]
        self.assertEqual(metadata["benchmark_domain_family_counts"], {"electronics": 1, "restaurant": 1, "telecom": 1})
        self.assertTrue(metadata["benchmark_domain_coverage_ok"])
        self.assertEqual(metadata["hardness_distribution"]["H3"], 1)
        self.assertEqual(metadata["hardness_distribution"]["H2"], 2)
        self.assertGreater(metadata["abstain_acceptable_rate"], 0.0)
        self.assertEqual(len(flat_rows), 3)

    def test_debug_benchmark_family_floor_restores_missing_core_family(self) -> None:
        rows_by_split = {
            "train": [
                {
                    "instance_id": "bench-elec",
                    "domain": "laptop",
                    "source_text": "Battery dies before evening.",
                    "domain_family": "electronics",
                    "group_id": "g1",
                    "gold_interpretations": [{"aspect_label": "battery", "sentiment": "negative", "evidence_text": "Battery dies before evening"}],
                    "implicit_grounded_interpretations": [{"aspect_label": "battery", "sentiment": "negative", "evidence_text": "Battery dies before evening"}],
                    "explicit_grounded_interpretations": [],
                    "hardness_tier": "H2",
                    "split_protocol": {"random": "train", "grouped": "train", "domain_holdout": "train"},
                }
            ],
            "val": [],
            "test": [],
        }
        fallback_rows_by_family = {
            "telecom": [
                {
                    "instance_id": "bench-tel",
                    "domain": "telecom",
                    "source_text": "Signal drops indoors.",
                    "domain_family": "telecom",
                    "group_id": "g2",
                    "gold_interpretations": [{"aspect_label": "connectivity", "sentiment": "negative", "evidence_text": "Signal drops indoors"}],
                    "implicit_grounded_interpretations": [{"aspect_label": "connectivity", "sentiment": "negative", "evidence_text": "Signal drops indoors"}],
                    "explicit_grounded_interpretations": [],
                    "hardness_tier": "H3",
                    "split_protocol": {"random": "train", "grouped": "train", "domain_holdout": "train"},
                    "split": "train",
                }
            ]
        }
        stats = _enforce_benchmark_family_floor(
            rows_by_split,
            source_domain_family_counts={"electronics": 1, "restaurant": 0, "telecom": 1},
            fallback_rows_by_family=fallback_rows_by_family,
            artifact_mode="debug_artifacts",
            seed=7,
        )
        self.assertTrue(stats["applied"])
        self.assertIn("telecom", stats["restored_families"])
        flat_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
        self.assertEqual({row["domain_family"] for row in flat_rows}, {"electronics", "telecom"})

    def test_train_floor_row_passes_for_grounded_non_general_row(self) -> None:
        row = {
            "domain": "laptop",
            "source_text": "Battery dies before evening.",
            "implicit": {
                "aspects": ["battery"],
                "review_reason": "weak_support",
                "spans": [{"support_type": "exact"}],
            },
        }
        self.assertTrue(
            _train_floor_row_passes(
                row,
                candidate_aspects_by_domain={},
                accepted_support_types={"exact", "near_exact", "gold"},
            )
        )

    def test_reasoned_strict_train_filter_excludes_borderline_rows(self) -> None:
        clean_row = {
            "id": "clean",
            "domain": "laptop",
            "implicit": {
                "needs_review": False,
                "review_reason": "none",
                "aspects": ["battery"],
                "spans": [{"support_type": "exact"}],
                "aspect_confidence": {"battery": 0.92},
            },
        }
        borderline_row = {
            "id": "borderline",
            "domain": "laptop",
            "implicit": {
                "needs_review": True,
                "review_reason": "weak_support",
                "aspects": ["battery"],
                "spans": [{"support_type": "exact"}],
                "aspect_confidence": {"battery": 0.74},
            },
        }
        kept_rows, soft_rows, hard_rows, stats = _split_train_review_filter(
            [clean_row, borderline_row],
            mode="reasoned_strict",
            candidate_aspects_by_domain={},
            min_confidence=0.8,
            accepted_support_types=("exact", "near_exact", "gold"),
        )

        self.assertEqual([row["id"] for row in kept_rows], ["clean"])
        self.assertEqual([row["id"] for row in soft_rows], ["borderline"])
        self.assertEqual(hard_rows, [])
        self.assertEqual(stats["train_review_rows_before_filter"], 2)
        self.assertEqual(stats["train_review_rows_after_filter"], 1)

    def test_quality_analysis_artifact_separates_borderline_and_rejected_rows(self) -> None:
        clean_row = {
            "id": "clean",
            "split": "train",
            "domain": "laptop",
            "implicit": {
                "needs_review": False,
                "review_reason": "none",
                "aspects": ["power"],
                "spans": [{"support_type": "exact"}],
                "aspect_confidence": {"power": 0.92},
            },
        }
        borderline_row = {
            "id": "borderline",
            "split": "train",
            "domain": "laptop",
            "implicit": {
                "needs_review": True,
                "review_reason": "low_confidence",
                "aspects": ["power"],
                "spans": [{"support_type": "exact"}],
                "aspect_confidence": {"power": 0.71},
            },
        }
        rejected_row = {
            "id": "rejected",
            "split": "train",
            "domain": "laptop",
            "implicit": {
                "needs_review": True,
                "review_reason": "implicit_not_ready",
                "aspects": ["general"],
                "spans": [],
                "aspect_confidence": {},
            },
        }
        artifact = _build_quality_analysis_artifact(
            [clean_row, borderline_row, rejected_row],
            [clean_row],
            min_confidence=0.8,
            accepted_support_types=("exact", "near_exact", "gold"),
        )

        self.assertEqual(artifact["summary"]["borderline_count"], 1)
        self.assertEqual(artifact["summary"]["rejected_count"], 1)
        self.assertEqual(artifact["borderline_rows"][0]["row"]["id"], "borderline")
        self.assertIn("low_confidence", artifact["borderline_rows"][0]["reason_codes"])
        self.assertEqual(artifact["rejected_rows"][0]["row"]["id"], "rejected")
        self.assertIn("implicit_not_ready", artifact["rejected_rows"][0]["reason_codes"])
        self.assertGreaterEqual(artifact["summary"]["reason_group_counts"].get("low_confidence", 0), 1)

    def test_analyze_reports_reads_actual_benchmark_artifacts(self) -> None:
        sys.path.insert(0, str(CODE_DIR))
        try:
            from analyze_reports import _build_scorecard  # noqa: E402
        finally:
            if str(CODE_DIR) in sys.path:
                sys.path.remove(str(CODE_DIR))

        output_dir = ROOT / "__tmp_out__" / "analyze_reports_test"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        benchmark_dir = output_dir / "benchmark" / "ambiguity_grounded"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_jsonl(
                benchmark_dir / "train.jsonl",
                [
                    {"instance_id": "a", "gold_interpretations": [{"aspect_label": "battery", "sentiment": "negative", "evidence_text": "dies"}]},
                    {"instance_id": "b", "gold_interpretations": [{"aspect_label": "screen", "sentiment": "positive", "evidence_text": "bright"}]},
                ],
            )
            write_jsonl(
                benchmark_dir / "val.jsonl",
                [
                    {"instance_id": "c", "gold_interpretations": [{"aspect_label": "sound", "sentiment": "positive", "evidence_text": "loud"}]},
                ],
            )
            write_jsonl(
                benchmark_dir / "test.jsonl",
                [
                    {"instance_id": "d", "gold_interpretations": [{"aspect_label": "keyboard", "sentiment": "negative", "evidence_text": "stiff"}]},
                ],
            )

            build = {
                "config": {"output_dir": str(output_dir), "train_max_positive_ratio": 0.5},
                "row_counts": {"selected": 4, "train_export": 4, "train": 2, "val": 1, "test": 1},
                "benchmark_artifact_counts": {"train": 0, "val": 0, "test": 0, "total": 0},
                "benchmark_summary": {"rows": 0, "split_counts": {"train": 0, "val": 0, "test": 0}},
                "benchmark_gold_eval": {
                    "has_gold_interpretations": True,
                    "num_rows_with_gold_interpretations": 4,
                    "average_gold_interpretations": 1.0,
                    "multi_gold_label_rate": 0.0,
                    "grounded_evidence_rate": 1.0,
                    "duplicate_interpretation_rate": 0.0,
                },
                "gold_eval": {"has_gold_labels": False},
                "output_quality": {
                    "fallback_only_rate": 0.1,
                    "needs_review_rows": 0,
                    "generic_implicit_aspects": 0,
                    "rejected_implicit_aspects": 0,
                    "domain_leakage_row_rate": 0.0,
                },
                "strict_quality": {
                    "explicit_in_implicit_rate": 0.0,
                    "boundary_false_positive_count": 0,
                    "h2_h3_ratio": 0.5,
                    "multi_aspect_ratio": 0.2,
                    "challenge_macro_f1": 0.6,
                },
                "train_general_dominance_rate": 0.0,
                "train_domain_leakage_row_rate": 0.0,
                "train_negative_ratio": 0.2,
                "train_positive_ratio": 0.2,
                "train_sentiment_constraints": {"achieved": {"neutral_ratio": 0.4}},
                "train_target_stats": {"size_within_target_range": True, "target_min_rows": 2, "target_max_rows": 10},
                "grounded_prediction_rate": 1.0,
                "ungrounded_non_general_count": 0,
                "unseen_domain_metrics": {
                    "unseen_non_general_coverage": 0.6,
                    "unseen_implicit_not_ready_rate": 0.1,
                    "unseen_domain_leakage_row_rate": 0.0,
                },
                "validation": {
                    "train_target_blocking_failure": False,
                    "sampled_run_blocked_or_debug": False,
                },
                "run_profile": "research",
                "artifact_mode": "research_release",
            }
            quality = build["output_quality"]
            scorecard = _build_scorecard(build, quality, None)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        self.assertEqual(scorecard["dataset_overview"]["benchmark_artifact_rows"], 4)
        self.assertFalse(scorecard["failure_analysis"]["artifact_parity"]["counts_match"])
        self.assertTrue(scorecard["overall_assessment"]["research_ready"])
        self.assertFalse(scorecard["overall_assessment"]["publication_ready"])

    def test_run_experiment_provider_resolution_uses_processor(self) -> None:
        self.assertEqual(_resolve_artifact_mode(run_profile="research", artifact_mode="auto"), "research_release")
        self.assertEqual(_resolve_artifact_mode(run_profile="debug", artifact_mode="auto"), "debug_artifacts")
        self.assertIsNone(resolve_processor_async_provider("local", model_name="llama-3.1-8b-instant"))
        self.assertIsNotNone(resolve_processor_async_provider("runpod", model_name="llama-3.1-8b-instant"))

    def test_dataset_builder_cli_accepts_core_optional_values(self) -> None:
        args = build_parser().parse_args(
            [
                "--run-profile",
                "debug",
                "--sample-size",
                "100",
                "--chunk-size",
                "25",
                "--chunk-offset",
                "5",
                "--llm-provider",
                "openai",
                "--llm-model-name",
                "gpt-4.1-mini",
                "--preview",
            ]
        )
        self.assertEqual(args.run_profile, "debug")
        self.assertEqual(args.sample_size, 100)
        self.assertEqual(args.chunk_size, 25)
        self.assertEqual(args.chunk_offset, 5)
        self.assertEqual(args.llm_provider, "openai")
        self.assertEqual(args.llm_model_name, "gpt-4.1-mini")
        self.assertTrue(args.preview)

    def test_dataset_builder_cli_supports_boolean_toggles(self) -> None:
        args = build_parser().parse_args(
            [
                "--no-use-coref",
                "--no-enable-llm-fallback",
                "--no-domain-conditioning",
                "--no-strict-implicit-enabled",
                "--no-progress",
                "--no-discovery-mode",
            ]
        )
        self.assertFalse(args.use_coref)
        self.assertFalse(args.enable_llm_fallback)
        self.assertFalse(args.use_domain_conditioning)
        self.assertFalse(args.strict_implicit_enabled)
        self.assertFalse(args.progress)
        self.assertFalse(args.discovery_mode)

    def test_grouped_split_keeps_non_empty_val_when_possible(self) -> None:
        rows = [
            {"id": "a1", "group": "g1"},
            {"id": "b1", "group": "g2"},
            {"id": "c1", "group": "g3"},
        ]
        train, val, test = grouped_split(
            rows,
            group_key="group",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        report = grouped_leakage_report(
            [{"split": "train", "group": r["group"]} for r in train]
            + [{"split": "val", "group": r["group"]} for r in val]
            + [{"split": "test", "group": r["group"]} for r in test],
            group_key="group",
        )
        self.assertEqual(report["overlap_counts"]["train_val"], 0)
        self.assertEqual(report["overlap_counts"]["train_test"], 0)
        self.assertEqual(report["overlap_counts"]["val_test"], 0)

    def test_benchmark_protocol_views_are_written(self) -> None:
        output_dir = ROOT / "__tmp_out__" / "protocol_export_test"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        benchmark_dir = output_dir / "benchmark" / "ambiguity_grounded"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows = [
                {
                    "id": "x1",
                    "split": "train",
                    "source_text": "Signal drops indoors",
                    "domain": "telecom",
                    "implicit": {"aspects": ["connectivity"], "hardness_tier": "H2", "spans": [{"latent_label": "connectivity", "support_type": "exact", "evidence_text": "Signal drops indoors"}]},
                    "gold_interpretations": [{"aspect_label": "connectivity", "sentiment": "negative", "evidence_text": "Signal drops indoors"}],
                },
                {
                    "id": "x2",
                    "split": "val",
                    "source_text": "Food was bland",
                    "domain": "restaurant",
                    "implicit": {"aspects": ["sensory quality"], "hardness_tier": "H2", "spans": [{"latent_label": "sensory quality", "support_type": "exact", "evidence_text": "Food was bland"}]},
                    "gold_interpretations": [{"aspect_label": "sensory quality", "sentiment": "negative", "evidence_text": "Food was bland"}],
                },
            ]
            assignments = {
                "x1": {"random": "train", "grouped": "train", "domain_holdout": "train"},
                "x2": {"random": "val", "grouped": "val", "domain_holdout": "val"},
            }
            from exporters import write_benchmark_outputs  # noqa: E402
            from build_dataset import _export_protocol_views  # noqa: E402
            by_split, meta, _ = _build_benchmark_instances(rows, assignments)
            protocol_views = _export_protocol_views(by_split)
            write_benchmark_outputs(benchmark_dir, by_split, meta, protocol_views=protocol_views)
            self.assertTrue((output_dir / "benchmark" / "random" / "train.jsonl").exists())
            self.assertTrue((output_dir / "benchmark" / "grouped" / "val.jsonl").exists())
            self.assertTrue((output_dir / "benchmark" / "domain_holdout" / "val.jsonl").exists())
            first_row = by_split["train"][0]
            self.assertIn("implicit_grounded_interpretations", first_row)
            self.assertIn("explicit_grounded_interpretations", first_row)
            if first_row["gold_interpretations"]:
                self.assertIn("domain_canonical_aspect", first_row["gold_interpretations"][0])
                self.assertIn("surface_rationale_tag", first_row["gold_interpretations"][0])
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_discover_aspects_is_deterministic_with_seed(self) -> None:
        rows = []
        for i in range(300):
            rows.append({"text": f"battery drains quickly {i}"})
        for i in range(300):
            rows.append({"text": f"service was rude {i}"})

        aspects_a = discover_aspects(rows, text_column="text", max_aspects=5, random_seed=42)
        aspects_b = discover_aspects(rows, text_column="text", max_aspects=5, random_seed=42)
        self.assertEqual(aspects_a, aspects_b)
        self.assertTrue(aspects_a)

    def test_discover_aspects_handles_tiny_sample_rate(self) -> None:
        rows = [{"text": f"battery drains quickly {i}"} for i in range(600)]
        aspects = discover_aspects(rows, text_column="text", max_aspects=3, sample_rate=0.0001, random_seed=123)
        self.assertTrue(aspects)

    def test_load_inputs_falls_back_to_serial_when_pool_fails(self) -> None:
        root = ROOT / "dataset_builder" / "tests" / "__tmp_inputs_fallback__"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        try:
            for i in range(3):
                (root / f"part{i}.csv").write_text("review\nworks fine\n", encoding="utf-8")

            class _BrokenPool:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    raise RuntimeError("pool unavailable")

                def __exit__(self, exc_type, exc, tb):
                    return False

            with patch("io_utils.ProcessPoolExecutor", _BrokenPool):
                frame = load_inputs(root)
            self.assertFalse(frame.empty)
            self.assertEqual(len(frame), 3)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_async_runpod_provider_does_not_forward_bypass_cache(self) -> None:
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"output": "ok"}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return _FakeResponse()

        provider = AsyncRunPodProvider(api_key="key123", base_url="https://api.runpod.ai/v2/x/run")
        with patch("llm_utils.httpx.AsyncClient", _FakeClient):
            res = asyncio.run(provider.generate("ping", "m", bypass_cache=True, temperature=0.2))

        self.assertEqual(res, "ok")
        self.assertNotIn("bypass_cache", captured["json"]["input"])

    def test_async_runpod_provider_refreshes_headers_after_env_bind(self) -> None:
        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"output": "ok"}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                captured["headers"] = headers
                return _FakeResponse()

        with patch.dict(os.environ, {}, clear=True):
            provider = AsyncRunPodProvider(api_key=None, base_url=None)
        with patch.dict(os.environ, {"REVIEWOP_RUNPOD_API_KEY": "late-key", "REVIEWOP_RUNPOD_ENDPOINT_URL": "https://api.runpod.ai/v2/x/run"}, clear=True):
            with patch("llm_utils.httpx.AsyncClient", _FakeClient):
                asyncio.run(provider.generate("ping", "m", bypass_cache=True))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer late-key")

    def test_smoke_probe_treats_error_string_as_failure(self) -> None:
        class _FailingProvider:
            async def generate(self, *args, **kwargs):
                return "Error: boom"

        cfg = BuilderConfig(
            input_dir=Path("__missing_input_dir__"),
            output_dir=Path("__tmp_out__"),
            processor="runpod",
            no_llm_cache=True,
            dry_run=True,
            preview_only=True,
            reset_output=False,
        )

        stdout = io.StringIO()
        with patch("llm_utils.resolve_async_llm_provider", return_value=_FailingProvider()):
            with patch("llm_utils.flush_llm_cache", return_value=None):
                with patch("implicit_pipeline.flush_llm_cache", return_value=None):
                    with patch("sys.stdout", stdout):
                        with self.assertRaises(ValueError):
                            asyncio.run(run_pipeline(cfg))
        text = stdout.getvalue()
        self.assertIn("Warning: RunPod connectivity probe failed", text)
        self.assertNotIn("RunPod Connectivity Verified.", text)

    def test_smoke_probe_prints_success_on_non_error(self) -> None:
        class _PassingProvider:
            async def generate(self, *args, **kwargs):
                return "pong"

        cfg = BuilderConfig(
            input_dir=Path("__missing_input_dir__"),
            output_dir=Path("__tmp_out__"),
            processor="runpod",
            no_llm_cache=True,
            dry_run=True,
            preview_only=True,
            reset_output=False,
        )

        stdout = io.StringIO()
        with patch("llm_utils.resolve_async_llm_provider", return_value=_PassingProvider()):
            with patch("llm_utils.flush_llm_cache", return_value=None):
                with patch("implicit_pipeline.flush_llm_cache", return_value=None):
                    with patch("sys.stdout", stdout):
                        with self.assertRaises(ValueError):
                            asyncio.run(run_pipeline(cfg))
        text = stdout.getvalue()
        self.assertIn("RunPod Connectivity Verified.", text)

    def test_strict_topup_recovery_reports_progress_for_candidate_screening(self) -> None:
        class _ProgressRecorder:
            def __init__(self) -> None:
                self.descriptions: list[str] = []
                self.updates: list[int] = []

            def set_description(self, label: str) -> None:
                self.descriptions.append(label)

            def update(self, n: int = 1) -> None:
                self.updates.append(n)

            def close(self) -> None:
                return None

        rows = [
            {
                "id": f"candidate-{idx}",
                "source_text": f"Battery case {idx}",
                "domain": "laptop",
                "implicit": {
                    "aspects": ["battery"],
                    "dominant_sentiment": "negative",
                    "needs_review": False,
                    "review_reason": "supported",
                    "spans": [{"support_type": "exact"}],
                    "aspect_confidence": {"battery": 0.9},
                },
            }
            for idx in range(3)
        ]
        progress = _ProgressRecorder()
        out_rows, stats = _strict_topup_recovery(
            train_rows=[],
            candidate_rows=rows,
            mode="strict_topup",
            target_min_rows=2,
            confidence_threshold=0.6,
            stage_b_confidence_threshold=0.55,
            stage_c_confidence_threshold=0.5,
            staged_recovery=True,
            allow_weak_support_in_stage_c=True,
            accepted_support_types=("exact", "near_exact", "gold"),
            candidate_aspects_by_domain={},
            seed=7,
            progress_bar=progress,
        )

        self.assertEqual(len(out_rows), 2)
        self.assertEqual(stats["topup_rows_added"], 2)
        self.assertGreaterEqual(len(progress.updates), 2)
        self.assertTrue(any(label.startswith("train export policies: topup recovery") for label in progress.descriptions))


if __name__ == "__main__":
    unittest.main()
