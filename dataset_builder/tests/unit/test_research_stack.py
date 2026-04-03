from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from research_stack import build_experiment_plan, build_research_manifest, default_benchmark_registry, default_model_registry, resolve_benchmark, resolve_model_family


class ResearchStackTests(unittest.TestCase):
    def test_registry_exposes_required_benchmarks_and_model_families(self) -> None:
        benchmarks = default_benchmark_registry()
        models = default_model_registry()
        self.assertIn("semeval_english_core", benchmarks)
        self.assertIn("shoes_acosi", benchmarks)
        self.assertIn("m_absa", benchmarks)
        self.assertIn("heuristic_latent", models)
        self.assertIn("zeroshot_latent", models)
        self.assertIn("supervised_ate", models)
        self.assertIn("hybrid_reasoner", models)
        self.assertIn("encoder_absa", models)
        self.assertIn("llm_prompted", models)

    def test_resolve_benchmark_prefers_dataset_family(self) -> None:
        benchmark = resolve_benchmark(domains=["laptop", "restaurant"])
        self.assertEqual(benchmark.key, "semeval_english_core")

        multilingual = resolve_benchmark(languages=["en", "de"])
        self.assertEqual(multilingual.key, "m_absa")

    def test_resolve_model_family_defaults_to_heuristic(self) -> None:
        model = resolve_model_family()
        self.assertEqual(model.key, "heuristic_latent")
        self.assertEqual(model.kind, "baseline")

    def test_build_experiment_plan_is_family_aware(self) -> None:
        plan = build_experiment_plan(benchmark_keys=["semeval_english_core", "shoes_acosi", "m_absa"], model_family_keys=["heuristic_latent", "encoder_absa", "end_to_end_absa", "llm_prompted", "augmentation"])
        keys = {(item.benchmark_key, item.model_family_key) for item in plan}
        self.assertIn(("semeval_english_core", "encoder_absa"), keys)
        self.assertIn(("shoes_acosi", "end_to_end_absa"), keys)
        self.assertIn(("m_absa", "augmentation"), keys)
        self.assertNotIn(("shoes_acosi", "augmentation"), keys)

    def test_research_manifest_contains_required_fields(self) -> None:
        benchmark = resolve_benchmark(domains=["laptop", "restaurant"])
        model = resolve_model_family("encoder_absa")
        manifest = build_research_manifest(
            dataset={"rows_in": 10, "rows_out": 10},
            benchmark=benchmark,
            model_family=model,
            metrics={"f1": 0.5},
        )
        self.assertEqual(manifest["benchmark"]["key"], "semeval_english_core")
        self.assertEqual(manifest["model_family"]["key"], "encoder_absa")
        self.assertEqual(manifest["metrics"]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
