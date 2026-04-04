from __future__ import annotations

import sys
from pathlib import Path
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from build_dataset import run_pipeline
from config import BuilderConfig
from utils import read_jsonl


def _write_fixture(path: Path, rows: list[tuple[str, int, str]]) -> None:
    lines = ["review_text,rating,domain"]
    for text, rating, domain in rows:
        lines.append(f'"{text}",{rating},{domain}')
    path.write_text("\n".join(lines), encoding="utf-8")


class CleanRoomPipelineTests(unittest.TestCase):
    def test_pipeline_supports_chunked_multi_domain_runs(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_clean_room"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        if tmp_root.exists():
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "explicit").mkdir(parents=True, exist_ok=True)
        (output_dir / "implicit").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "stale.txt").write_text("old-data", encoding="utf-8")

        _write_fixture(
            input_dir / "restaurant.csv",
            [
                ("Great food and friendly service, but the table spacing felt cramped and noisy.", 5, "restaurant"),
                ("Slow service but delicious dessert; billing was transparent and fair.", 3, "restaurant"),
                ("The waiter was helpful, yet the soup was cold while the pasta was excellent.", 4, "restaurant"),
            ],
        )
        _write_fixture(
            input_dir / "laptop.csv",
            [
                ("Bright screen and fast performance, although the fan gets noisy under load.", 5, "laptop"),
                ("Keyboard feels cheap but battery is good and charging is quick.", 3, "laptop"),
                ("Trackpad is responsive, but the speakers distort at high volume.", 4, "laptop"),
            ],
        )
        _write_fixture(
            input_dir / "service.csv",
            [
                ("Support was quick and polite, but resolution required two follow ups.", 5, "service"),
                ("The app keeps crashing and login security prompts are confusing.", 1, "service"),
                ("Help desk answered fast while chat notifications remained delayed.", 4, "service"),
            ],
        )
        _write_fixture(
            input_dir / "hotel.csv",
            [
                ("Room was clean and spacious, but check-in was slow and confusing.", 4, "hotel"),
                ("Great breakfast and comfortable bed; elevator noise was annoying at night.", 3, "hotel"),
                ("Staff communication was clear, though air conditioning failed twice.", 2, "hotel"),
            ],
        )
        _write_fixture(
            input_dir / "finance.csv",
            [
                ("Loan pricing was fair, but the app navigation was confusing and slow.", 3, "finance"),
                ("Customer support explained fees clearly; approval time was excellent.", 5, "finance"),
                ("Statement transparency improved, yet fraud alerts arrived too late.", 2, "finance"),
            ],
        )
        _write_fixture(
            input_dir / "healthcare.csv",
            [
                ("Nurse communication was kind, but waiting time was very long.", 3, "healthcare"),
                ("Doctor explanation was excellent and follow-up reminders were timely.", 5, "healthcare"),
                ("Clinic cleanliness was good while billing support remained unclear.", 4, "healthcare"),
            ],
        )

        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            sample_size=18,
            chunk_size=6,
            chunk_offset=0,
            run_profile="debug",
            min_text_tokens=2,
            evaluation_protocol="loo",
            domain_holdout="hotel",
            train_review_filter_mode="keep",
            train_fallback_general_policy="keep",
            confidence_threshold=0.0,
        )
        report = run_pipeline(cfg)

        self.assertEqual(report["pipeline_version"], "5.5-production")
        self.assertEqual(report["split_sizes"]["train"] + report["split_sizes"]["val"] + report["split_sizes"]["test"], 6)
        self.assertTrue((output_dir / "explicit" / "train.jsonl").exists())
        self.assertTrue((output_dir / "implicit" / "train.jsonl").exists())
        self.assertTrue((output_dir / "implicit_strict" / "train.jsonl").exists())
        self.assertTrue((output_dir / "implicit_strict" / "review_queue.jsonl").exists())
        self.assertTrue((output_dir / "implicit_strict" / "challenge.jsonl").exists())
        self.assertTrue((output_dir / "reports" / "build_report.json").exists())
        self.assertTrue((output_dir / "compat" / "protonet" / "reviewlevel" / "train.jsonl").exists())
        self.assertFalse((output_dir / "stale.txt").exists())

        explicit_rows = read_jsonl(output_dir / "explicit" / "train.jsonl")
        if explicit_rows:
            self.assertNotIn("aspect", explicit_rows[0])
            self.assertNotIn("polarity", explicit_rows[0])

        implicit_rows = read_jsonl(output_dir / "implicit" / "train.jsonl")
        if not implicit_rows:
            implicit_rows = read_jsonl(output_dir / "implicit" / "val.jsonl") + read_jsonl(output_dir / "implicit" / "test.jsonl")
        domains = {row["domain"] for row in implicit_rows}
        self.assertTrue({"restaurant", "laptop", "service", "hotel", "finance", "healthcare"} & domains)
        self.assertTrue(all("latent_label" in span and "hardness_tier" in span for row in implicit_rows for span in row["implicit"]["spans"]))

        report = json.loads((output_dir / "reports" / "build_report.json").read_text(encoding="utf-8"))
        quality = report["output_quality"]
        data_quality = json.loads((output_dir / "reports" / "data_quality_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["implicit_mode"], "zeroshot")
        self.assertIn("language_distribution", report)
        self.assertIn("candidate_aspects_by_language", report)
        self.assertIn("chunk_preview_size", report)
        self.assertIn("row_counts", report)
        self.assertIn("validation", report)
        self.assertTrue(report["validation"]["counts_match"])
        self.assertIn("grounded_prediction_rate", report)
        self.assertIn("ungrounded_non_general_count", report)
        self.assertIn("train_general_rows_before_policy", report)
        self.assertIn("train_general_rows_after_policy", report)
        self.assertIn("train_general_policy_applied", report)
        self.assertIn("train_sentiment_before_balance", report)
        self.assertIn("train_sentiment_after_balance", report)
        self.assertIn("train_general_dominance_rate", report)
        self.assertIn("train_domain_leakage_row_rate", report)
        self.assertIn("train_salvage_stats", report)
        self.assertIn("train_topup_stats", report)
        self.assertIn("train_topup_rejection_breakdown", report)
        self.assertIn("topup_effectiveness", report)
        self.assertIn("size_recovery_stage", report)
        self.assertIn("size_recovery_shortfall_remaining", report)
        self.assertIn("train_sentiment_constraints", report)
        self.assertIn("train_target_stats", report)
        self.assertIn("train_positive_ratio", report)
        self.assertIn("eval_domain_leakage_row_rate", report)
        self.assertIn("train_export", report["row_counts"])
        self.assertIn("gold_eval", report)
        self.assertFalse(report["gold_eval"]["has_gold_labels"])
        self.assertIn("domain_generalization", report)
        self.assertIn("leave_one_domain_out", report["domain_generalization"])
        self.assertIn("novelty_identity", report)
        self.assertTrue(report["novelty_identity"]["hybrid_explicit_implicit_pipeline"])
        self.assertTrue(report["novelty_identity"]["structured_fallback_taxonomy_present"])
        self.assertIn("research", report)
        self.assertIn(report["research"]["benchmark_family"], {"english_core", "implicit_heavy", "multilingual", "auxiliary"})
        self.assertEqual(data_quality["implicit_mode"], "zeroshot")
        self.assertIn("multilingual_mode", data_quality)
        self.assertIn("coreference_enabled", data_quality)
        self.assertIn("language_distribution", data_quality)
        self.assertIn("row_counts", data_quality)
        self.assertIn("research", data_quality)
        self.assertIn("train_salvage_stats", data_quality)
        self.assertIn("train_topup_stats", data_quality)
        self.assertIn("train_target_stats", data_quality)
        self.assertIn("train_positive_ratio", data_quality)
        self.assertIn("strict_quality", data_quality)
        self.assertIn("strict_artifacts", data_quality)
        self.assertEqual(quality["fallback_only_rows"], data_quality["output_quality"]["fallback_only_rows"])
        self.assertEqual(quality["span_support"], data_quality["output_quality"]["span_support"])
        self.assertEqual(report["implicit_diagnostics"]["fallback_only_count"], quality["fallback_only_rows"])
        self.assertEqual(report["implicit_diagnostics"]["span_support"]["exact"], quality["span_support"]["exact"])
        self.assertEqual(report["implicit_diagnostics"]["span_support"]["near_exact"], quality["span_support"]["near_exact"])
        self.assertEqual(quality["generic_implicit_aspects"], 0)
        self.assertEqual(quality["rejected_implicit_aspects"], 0)
        self.assertEqual(report["train_general_dominance_rate"], 0.0)
        self.assertEqual(report["train_domain_leakage_row_rate"], 0.0)
        self.assertTrue(report["validation"]["train_domain_leakage_ok"])
        self.assertEqual(report["run_profile"], "debug")
        self.assertEqual(report["promotion_eligibility"], "blocked_debug")
        self.assertIn("top_implicit_aspects_by_split", quality)
        self.assertIn("top_implicit_aspects_by_domain", quality)
        self.assertIn("fallback_only_rate_by_split", quality)
        self.assertIn("span_support", quality)
        self.assertIn("review_reason_counts", quality)
        self.assertIn("fallback_branch_counts", quality)
        self.assertIn("strict_quality", quality)
        self.assertIn("explicit_in_implicit_rate", quality)
        self.assertIn("h2_h3_ratio", quality)
        self.assertIn("multi_aspect_ratio", quality)
        self.assertIn("review_reason_counts", report["implicit_diagnostics"])
        self.assertIn("fallback_branch_counts", report["implicit_diagnostics"])

        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_pipeline_merges_gold_annotations_and_reports_gold_eval(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_gold_eval"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        if tmp_root.exists():
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "explicit").mkdir(parents=True, exist_ok=True)
        (output_dir / "implicit").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)

        text = "Great food and friendly service"
        _write_fixture(
            input_dir / "restaurant.csv",
            [
                (text, 5, "restaurant"),
                ("Slow service but delicious dessert", 3, "restaurant"),
                ("The waiter was helpful", 4, "restaurant"),
            ],
        )
        gold_path = input_dir / "gold_annotations.jsonl"
        gold_row = {
            "record_id": None,
            "domain": "restaurant",
            "text": text,
            "gold_labels": [{"aspect": "food quality", "sentiment": "positive", "start": 0, "end": 10}],
            "annotator_id": "annotator_1",
            "review_status": "approved",
        }
        gold_path.write_text(json.dumps(gold_row) + "\n", encoding="utf-8")

        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            min_text_tokens=2,
            gold_annotations_path=gold_path,
            emit_review_set=True,
            review_set_size=2,
        )
        report = run_pipeline(cfg)
        self.assertIn("gold_eval", report)
        self.assertTrue(report["gold_eval"]["has_gold_labels"])
        self.assertGreaterEqual(report["gold_eval"]["num_rows_with_gold"], 1)
        self.assertIn("by_domain", report["gold_eval"])
        self.assertIn("grounded_prediction_rate", report)
        self.assertTrue((output_dir / "reports" / "review_set_template.jsonl").exists())

        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_adaptive_conditioning_improves_unseen_domain_coverage_vs_strict(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_unseen_compare"
        input_dir = tmp_root / "input"
        output_soft = tmp_root / "output_soft"
        output_strict = tmp_root / "output_strict"
        if tmp_root.exists():
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        _write_fixture(input_dir / "fashion.csv", [
            ("The fabric is great but stitching is poor.", 3, "fashion"),
            ("Return support was slow and confusing.", 2, "fashion"),
            ("Comfort is excellent for daily wear.", 5, "fashion"),
            ("Price feels high for this quality.", 2, "fashion"),
            ("Delivery was fast and packaging neat.", 4, "fashion"),
            ("The zipper broke after a week.", 1, "fashion"),
        ])
        _write_fixture(input_dir / "banking.csv", [
            ("App login is smooth and reliable.", 5, "banking"),
            ("Transaction alerts are delayed.", 2, "banking"),
            ("Customer support explained fees clearly.", 4, "banking"),
            ("The app crashes during payment.", 1, "banking"),
            ("KYC flow is easy to navigate.", 4, "banking"),
            ("Call center wait time is too long.", 2, "banking"),
        ])
        _write_fixture(input_dir / "gaming.csv", [
            ("Gameplay is responsive and smooth.", 5, "gaming"),
            ("Server lag ruins ranked matches.", 1, "gaming"),
            ("Matchmaking is slow at night.", 2, "gaming"),
            ("Graphics quality is excellent.", 5, "gaming"),
            ("Client crashes after update.", 1, "gaming"),
            ("Tutorial helps new players quickly.", 4, "gaming"),
        ])
        _write_fixture(input_dir / "telecom.csv", [
            ("Network speed is great outdoors.", 5, "telecom"),
            ("Call drops happen frequently indoors.", 2, "telecom"),
            ("Billing included unexpected fees.", 1, "telecom"),
            ("Support resolved SIM issue quickly.", 4, "telecom"),
            ("The app UI is slow and cluttered.", 2, "telecom"),
            ("SMS OTP delivery is reliable.", 4, "telecom"),
        ])

        soft_cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_soft,
            text_column_override="review_text",
            min_text_tokens=2,
            run_profile="research",
            domain_conditioning_mode="adaptive_soft",
            weak_domain_support_row_threshold=80,
        )
        strict_cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_strict,
            text_column_override="review_text",
            min_text_tokens=2,
            run_profile="research",
            domain_conditioning_mode="strict_hard",
            strict_domain_conditioning=True,
            weak_domain_support_row_threshold=80,
        )

        soft_report = run_pipeline(soft_cfg)
        strict_report = run_pipeline(strict_cfg)

        soft_unseen = soft_report["unseen_domain_metrics"]
        strict_unseen = strict_report["unseen_domain_metrics"]
        self.assertEqual(soft_report["train_domain_conditioning_mode"], "strict_hard")
        self.assertEqual(soft_report["eval_domain_conditioning_mode"], "adaptive_soft")
        self.assertEqual(soft_report["train_domain_leakage_row_rate"], 0.0)
        self.assertEqual(strict_report["train_domain_leakage_row_rate"], 0.0)
        self.assertGreaterEqual(soft_unseen["unseen_non_general_coverage"], strict_unseen["unseen_non_general_coverage"])
        self.assertLessEqual(soft_unseen["unseen_implicit_not_ready_rate"], strict_unseen["unseen_implicit_not_ready_rate"])
        self.assertEqual(soft_report["train_general_dominance_rate"], 0.0)

        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_debug_sampled_run_marked_non_promotable(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_debug_sampled_contract"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        if tmp_root.exists():
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_fixture(input_dir / "restaurant.csv", [
            ("Great food and friendly service.", 5, "restaurant"),
            ("Service was slow and billing confusing.", 2, "restaurant"),
            ("Tasty dessert but noisy seating.", 3, "restaurant"),
            ("Quick staff response and clean table.", 4, "restaurant"),
        ])
        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            sample_size=20,
            chunk_size=4,
            run_profile="debug",
            min_text_tokens=2,
        )
        report = run_pipeline(cfg)
        self.assertEqual(report["run_profile"], "debug")
        self.assertEqual(report["promotion_eligibility"], "blocked_debug")
        self.assertTrue(report["validation"]["sampled_run_blocked_or_debug"])

        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
