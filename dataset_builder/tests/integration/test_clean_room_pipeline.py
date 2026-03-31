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
                ("Great food and friendly service", 5, "restaurant"),
                ("Slow service but delicious dessert", 3, "restaurant"),
                ("The waiter was helpful", 4, "restaurant"),
            ],
        )
        _write_fixture(
            input_dir / "laptop.csv",
            [
                ("Bright screen and fast performance", 5, "laptop"),
                ("Keyboard feels cheap but battery is good", 3, "laptop"),
                ("Trackpad is responsive", 4, "laptop"),
            ],
        )
        _write_fixture(
            input_dir / "service.csv",
            [
                ("Support was quick and polite", 5, "service"),
                ("The app keeps crashing", 1, "service"),
                ("Help desk answered fast", 4, "service"),
            ],
        )

        cfg = BuilderConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column_override="review_text",
            sample_size=6,
            chunk_size=2,
            chunk_offset=0,
            min_text_tokens=2,
        )
        report = run_pipeline(cfg)

        self.assertEqual(report["pipeline_version"], "1.0-clean-room")
        self.assertEqual(report["split_sizes"]["train"] + report["split_sizes"]["val"] + report["split_sizes"]["test"], 2)
        self.assertEqual(report["chunk_preview_size"], 1)
        self.assertEqual(report["chunk_sampling_strategy"], "seeded_shuffle_then_slice")
        self.assertTrue((output_dir / "explicit" / "train.jsonl").exists())
        self.assertTrue((output_dir / "implicit" / "train.jsonl").exists())
        self.assertTrue((output_dir / "reports" / "build_report.json").exists())
        self.assertTrue((output_dir / "compat" / "protonet" / "reviewlevel" / "train.jsonl").exists())
        self.assertFalse((output_dir / "stale.txt").exists())

        explicit_rows = read_jsonl(output_dir / "explicit" / "train.jsonl")
        self.assertTrue(explicit_rows)
        self.assertNotIn("aspect", explicit_rows[0])
        self.assertNotIn("polarity", explicit_rows[0])

        implicit_rows = read_jsonl(output_dir / "implicit" / "train.jsonl")
        domains = {row["domain"] for row in implicit_rows}
        self.assertTrue({"restaurant", "laptop", "service"} & domains)
        self.assertTrue(all("latent_aspect" in span and "surface_aspect" in span for row in implicit_rows for span in row["implicit"]["spans"]))

        report = json.loads((output_dir / "reports" / "build_report.json").read_text(encoding="utf-8"))
        quality = report["output_quality"]
        data_quality = json.loads((output_dir / "reports" / "data_quality_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["implicit_mode"], "heuristic")
        self.assertIn("research", report)
        self.assertIn(report["research"]["benchmark_family"], {"english_core", "implicit_heavy", "multilingual", "auxiliary"})
        self.assertEqual(data_quality["implicit_mode"], "heuristic")
        self.assertIn("research", data_quality)
        self.assertEqual(quality["fallback_only_rows"], data_quality["output_quality"]["fallback_only_rows"])
        self.assertEqual(quality["span_support"], data_quality["output_quality"]["span_support"])
        self.assertEqual(report["implicit_diagnostics"]["fallback_only_count"], quality["fallback_only_rows"])
        self.assertEqual(report["implicit_diagnostics"]["span_support"], quality["span_support"])
        self.assertEqual(quality["generic_implicit_aspects"], 0)
        self.assertEqual(quality["rejected_implicit_aspects"], 0)
        self.assertIn("top_implicit_aspects_by_split", quality)
        self.assertIn("top_implicit_aspects_by_domain", quality)
        self.assertIn("fallback_only_rate_by_split", quality)
        self.assertIn("span_support", quality)

        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
