from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from analyze_reports import main as analyze_main
from build_dataset import main as build_main, run_pipeline
from contracts import BuilderConfig


def _write_fixture(path: Path, rows: list[tuple[str, int]]) -> None:
    lines = ["review_text,rating"]
    for text, rating in rows:
        lines.append(f'"{text}",{rating}')
    path.write_text("\n".join(lines), encoding="utf-8")


class RunProfileAndAnalysisTests(unittest.TestCase):
    def test_research_profile_rejects_sampled_runs(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_profile_research_sampled"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_fixture(input_dir / "restaurant.csv", [("Great food and service", 5), ("Slow billing", 2)])
        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            sample_size=10,
            chunk_size=5,
            run_profile="research",
        )
        with self.assertRaises(ValueError):
            run_pipeline(cfg)
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_debug_profile_allows_sampled_runs_but_marks_non_promotable(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_profile_debug_sampled"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_fixture(input_dir / "restaurant.csv", [("Great food and service", 5), ("Slow billing", 2), ("Friendly staff", 4)])
        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            sample_size=10,
            chunk_size=3,
            run_profile="debug",
        )
        report = run_pipeline(cfg)
        self.assertEqual(report["run_profile"], "debug")
        self.assertEqual(report["promotion_eligibility"], "blocked_debug")
        self.assertTrue(report["validation"]["sampled_run_blocked_or_debug"])
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_research_profile_size_shortfall_blocks_main_exit_code(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_profile_size_block"
        input_dir = tmp_root / "input"
        output_dir = tmp_root / "output"
        shutil.rmtree(tmp_root, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        _write_fixture(input_dir / "restaurant.csv", [("Great food and service", 5), ("Slow billing", 2), ("Friendly staff", 4)])
        exit_code = build_main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--text-column",
                "review_text",
                "--run-profile",
                "research",
            ]
        )
        self.assertEqual(exit_code, 2)
        report = json.loads((output_dir / "reports" / "build_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["validation"]["train_target_blocking_failure"])
        shutil.rmtree(tmp_root, ignore_errors=True)

    def test_analyzer_outputs_scorecard_and_verdict(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        tmp_root = repo_root / "dataset_builder" / "output" / "_tmp_test_analysis_scorecard"
        reports_dir = tmp_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        build_report = {
            "run_profile": "research",
            "row_counts": {"selected": 25, "train_export": 8},
            "output_quality": {"fallback_only_rate": 0.28, "needs_review_rows": 10, "domain_leakage_row_rate": 0.12},
            "train_domain_leakage_row_rate": 0.0,
            "train_general_dominance_rate": 0.0,
            "grounded_prediction_rate": 1.0,
            "train_target_stats": {"target_min_rows": 2200, "size_within_target_range": False, "size_shortfall_rows": 2192},
            "validation": {
                "train_target_blocking_failure": True,
                "train_general_excluded": True,
                "train_domain_leakage_ok": True,
                "no_generic_aspects": True,
                "no_rejected_aspects": True,
            },
            "gold_eval": {"has_gold_labels": False},
        }
        data_quality = {"output_quality": {"fallback_only_rate": 0.28}}
        prev_report = {
            "row_counts": {"selected": 20, "train_export": 7},
            "output_quality": {"fallback_only_rate": 0.30, "domain_leakage_row_rate": 0.14},
        }
        (reports_dir / "build_report.json").write_text(json.dumps(build_report), encoding="utf-8")
        (reports_dir / "data_quality_report.json").write_text(json.dumps(data_quality), encoding="utf-8")
        (reports_dir / "build_report.previous.json").write_text(json.dumps(prev_report), encoding="utf-8")
        exit_code = analyze_main(
            [
                "--build-report",
                str(reports_dir / "build_report.json"),
                "--data-quality-report",
                str(reports_dir / "data_quality_report.json"),
                "--previous-build-report",
                str(reports_dir / "build_report.previous.json"),
            ]
        )
        self.assertEqual(exit_code, 0)
        scorecard = json.loads((reports_dir / "deep_analysis_report.json").read_text(encoding="utf-8"))
        self.assertIn("major_improvements", scorecard)
        self.assertIn("critical_failures", scorecard)
        self.assertIn("quality_deltas_vs_previous", scorecard)
        self.assertIn("verdict", scorecard)
        self.assertFalse(scorecard["verdict"]["research_ready"])
        self.assertTrue((reports_dir / "deep_analysis_report.md").exists())
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
