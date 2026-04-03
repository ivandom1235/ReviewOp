from __future__ import annotations

import sys
from pathlib import Path
import unittest
import uuid
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from contracts import BuilderConfig
from build_dataset import build_parser as build_dataset_parser, _resolve_split_domain_conditioning_modes


class BuilderConfigTests(unittest.TestCase):
    def test_split_domain_modes_default_to_train_strict_eval_adaptive(self) -> None:
        cfg = BuilderConfig()
        train_mode, eval_mode = _resolve_split_domain_conditioning_modes(cfg)
        self.assertEqual(train_mode, "strict_hard")
        self.assertEqual(eval_mode, "adaptive_soft")

    def test_split_domain_modes_honor_legacy_strict_when_split_modes_not_overridden(self) -> None:
        cfg = BuilderConfig(
            strict_domain_conditioning=True,
            domain_conditioning_mode="adaptive_soft",
            train_domain_conditioning_mode="strict_hard",
            eval_domain_conditioning_mode="adaptive_soft",
        )
        train_mode, eval_mode = _resolve_split_domain_conditioning_modes(cfg)
        self.assertEqual(train_mode, "strict_hard")
        self.assertEqual(eval_mode, "strict_hard")

    def test_ensure_dirs_skips_reset_for_preview_and_dry_run(self) -> None:
        root = Path(__file__).resolve().parents[3] / "dataset_builder" / "output" / f"_tmp_test_contracts_{uuid.uuid4().hex}"
        output_dir = root / "output"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            stale_file = output_dir / "stale.txt"
            stale_file.write_text("keep-me", encoding="utf-8")

            cfg = BuilderConfig(output_dir=output_dir, dry_run=True, preview_only=True, reset_output=True)
            cfg.ensure_dirs()

            self.assertTrue(stale_file.exists())
            self.assertTrue((output_dir / "explicit").exists())
            self.assertTrue((output_dir / "implicit").exists())
            self.assertTrue((output_dir / "reports").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_ensure_dirs_resets_output_for_write_runs(self) -> None:
        root = Path(__file__).resolve().parents[3] / "dataset_builder" / "output" / f"_tmp_test_contracts_{uuid.uuid4().hex}"
        output_dir = root / "output"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            stale_file = output_dir / "stale.txt"
            stale_file.write_text("remove-me", encoding="utf-8")

            cfg = BuilderConfig(output_dir=output_dir, dry_run=False, preview_only=False, reset_output=True)
            cfg.ensure_dirs()

            self.assertFalse(stale_file.exists())
            self.assertTrue((output_dir / "explicit").exists())
            self.assertTrue((output_dir / "implicit").exists())
            self.assertTrue((output_dir / "reports").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_build_dataset_parser_supports_llm_fallback_inverse_flag(self) -> None:
        parser = build_dataset_parser()
        defaults = parser.parse_args([])
        self.assertTrue(defaults.enable_llm_fallback)
        self.assertEqual(defaults.train_fallback_general_policy, "cap")
        self.assertEqual(defaults.train_review_filter_mode, "reasoned_strict")
        self.assertEqual(defaults.train_salvage_mode, "recover_non_general")
        self.assertEqual(defaults.train_sentiment_balance_mode, "cap_neutral_with_dual_floor")
        self.assertEqual(defaults.train_topup_recovery_mode, "strict_topup")
        self.assertTrue(defaults.train_topup_staged_recovery)
        self.assertTrue(defaults.train_topup_allow_weak_support_in_stage_c)
        self.assertAlmostEqual(defaults.train_topup_stage_b_confidence_threshold, 0.54)
        self.assertAlmostEqual(defaults.train_topup_stage_c_confidence_threshold, 0.52)
        self.assertAlmostEqual(defaults.train_max_positive_ratio, 0.5)
        self.assertAlmostEqual(defaults.train_neutral_max_ratio, 0.58)
        self.assertFalse(defaults.strict_domain_conditioning)
        self.assertEqual(defaults.domain_conditioning_mode, "adaptive_soft")
        self.assertEqual(defaults.run_profile, "research")
        self.assertIsNone(defaults.train_domain_conditioning_mode)
        self.assertIsNone(defaults.eval_domain_conditioning_mode)
        self.assertTrue(defaults.progress)
        debug = parser.parse_args(["--run-profile", "debug"])
        self.assertEqual(debug.run_profile, "debug")
        no_progress = parser.parse_args(["--no-progress"])
        self.assertFalse(no_progress.progress)
        disabled = parser.parse_args(["--no-enable-llm-fallback"])
        self.assertFalse(disabled.enable_llm_fallback)


if __name__ == "__main__":
    unittest.main()
