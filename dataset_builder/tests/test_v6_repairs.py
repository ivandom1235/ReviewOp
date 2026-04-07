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

from implicit_pipeline import build_implicit_row  # noqa: E402
from implicit_pipeline import discover_aspects  # noqa: E402
from build_dataset import _build_benchmark_instances, _group_identity  # noqa: E402
from build_dataset import run_pipeline  # noqa: E402
from contracts import BuilderConfig  # noqa: E402
from evaluation import benchmark_gold_eval, gold_eval  # noqa: E402
from io_utils import load_inputs  # noqa: E402
from llm_utils import AsyncRunPodProvider  # noqa: E402
from run_experiment import _resolve_artifact_mode, _resolve_llm_provider  # noqa: E402
from splitter import grouped_leakage_report, grouped_split  # noqa: E402
from utils import write_jsonl  # noqa: E402


class V6RepairTests(unittest.TestCase):
    def test_group_identity_prefers_entity_keys_over_unique_id(self) -> None:
        row = {"id": "unique-row-id", "product_id": "SKU-123", "domain": "laptop", "source_file": "x.csv"}
        self.assertEqual(_group_identity(row), "sku-123")

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

    def test_benchmark_dedup_and_grounding(self) -> None:
        row = {
            "id": "r3",
            "split": "train",
            "source_text": "Battery dies before evening and gets very hot.",
            "domain": "laptop",
            "gold_interpretations": [
                {"aspect": "power", "sentiment": "negative", "evidence": "Battery dies before evening", "annotator_support": 1},
                {"aspect": "power", "sentiment": "negative", "evidence": "Battery dies before evening", "annotator_support": 1},
                {"aspect": "thermal", "sentiment": "negative", "evidence": "hot", "annotator_support": 1},
                {"aspect": "thermal", "sentiment": "negative", "evidence": "ice", "annotator_support": 1},
            ],
            "implicit": {"aspects": ["power", "thermal"], "aspect_confidence": {"power": 0.9, "thermal": 0.8}, "spans": []},
        }
        rows_by_split, metadata = _build_benchmark_instances([row], {"r3": {"random": "train", "source_holdout": "train", "domain_holdout": "train"}})
        populated_split = next(split for split in ("train", "val", "test") if rows_by_split.get(split))
        out = rows_by_split[populated_split][0]
        interpretations = out["gold_interpretations"]
        # Duplicate power interpretation is collapsed and ungrounded "ice" is removed.
        self.assertEqual(len(interpretations), 2)
        power = next(item for item in interpretations if item["aspect"] == "power")
        self.assertEqual(int(power["annotator_support"]), 2)
        self.assertGreaterEqual(float(metadata["grounded_evidence_rate"]), 0.6)

    def test_benchmark_gold_eval_tracks_benchmark_interpretations(self) -> None:
        rows = [
            {
                "id": "bench-1",
                "domain": "laptop",
                "review_text": "Battery dies before evening.",
                "gold_interpretations": [
                    {"aspect": "battery", "sentiment": "negative", "evidence": "Battery dies"},
                    {"aspect": "battery", "sentiment": "negative", "evidence": "Battery dies"},
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
            assignments[row_id] = {"random": "train", "source_holdout": "train", "domain_holdout": "train"}
        benchmark_rows_by_split, metadata = _build_benchmark_instances(
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

    def test_analyze_reports_reads_actual_benchmark_artifacts(self) -> None:
        from analyze_reports import _build_scorecard  # noqa: E402

        output_dir = ROOT / "__tmp_out__" / "analyze_reports_test"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        benchmark_dir = output_dir / "benchmark" / "ambiguity_openworld"
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_jsonl(
                benchmark_dir / "train.jsonl",
                [
                    {"instance_id": "a", "gold_interpretations": [{"aspect": "battery", "sentiment": "negative", "evidence": "dies"}]},
                    {"instance_id": "b", "gold_interpretations": [{"aspect": "screen", "sentiment": "positive", "evidence": "bright"}]},
                ],
            )
            write_jsonl(
                benchmark_dir / "val.jsonl",
                [
                    {"instance_id": "c", "gold_interpretations": [{"aspect": "sound", "sentiment": "positive", "evidence": "loud"}]},
                ],
            )
            write_jsonl(
                benchmark_dir / "test.jsonl",
                [
                    {"instance_id": "d", "gold_interpretations": [{"aspect": "keyboard", "sentiment": "negative", "evidence": "stiff"}]},
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

    def test_run_experiment_provider_resolution_prefers_runpod_for_release(self) -> None:
        self.assertEqual(_resolve_artifact_mode(run_profile="research", artifact_mode="auto"), "research_release")
        self.assertEqual(_resolve_artifact_mode(run_profile="debug", artifact_mode="auto"), "debug_artifacts")
        self.assertEqual(_resolve_llm_provider(llm_provider="auto", artifact_mode="research_release"), "runpod")
        self.assertEqual(_resolve_llm_provider(llm_provider="auto", artifact_mode="debug_artifacts"), "mock")

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
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "late-key", "RUNPOD_ENDPOINT_URL": "https://api.runpod.ai/v2/x/run"}, clear=True):
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
            llm_provider="mock",
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
            llm_provider="mock",
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


if __name__ == "__main__":
    unittest.main()
