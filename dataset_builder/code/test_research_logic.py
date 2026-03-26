from __future__ import annotations

from pathlib import Path
import shutil

from build_dataset import run_pipeline
from config import BuilderConfig
from utils import read_jsonl


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scratch_root = repo_root / "dataset_builder" / "output" / "regression_smoke"
    input_dir = scratch_root / "input"
    output_dir = scratch_root / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = input_dir / "sample.csv"
    sample_csv.write_text(
        "review_text,rating,brand\n"
        "The sauce is delicious and the steak was excellent,5,Bistro\n"
        "The service was a bit slow but the food was good,3,Bistro\n"
        "Garageband keeps crashing during playback,1,Acme\n"
        "Battery life is good and the pointer is not moving,2,Acme\n"
        "The screen is bright but the performance is too slow,3,Acme\n"
        "Mine was a little burnt and never really worked,1,Bistro\n",
        encoding="utf-8",
    )

    cfg = BuilderConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        text_column_override="review_text",
        enable_llm_fallback=False,
    )
    report = run_pipeline(cfg)

    assert report["implicit_diagnostics"]["explicit_leakage_count"] >= 0
    assert report["implicit_diagnostics"]["accepted_clause_count"] >= 0
    assert "confusion_patterns" in report["implicit_diagnostics"]

    implicit_rows = read_jsonl(output_dir / "implicit" / "train.jsonl") + read_jsonl(output_dir / "implicit" / "val.jsonl") + read_jsonl(output_dir / "implicit" / "test.jsonl")
    joined = "\n".join(row["source_text"] for row in implicit_rows)
    assert "Garageband keeps crashing during playback" in joined

    for row in implicit_rows:
        text = row["source_text"].lower()
        implicit = row["implicit"]
        aspects = set(implicit["aspects"])
        if "garageband keeps crashing during playback" in text:
            assert "performance" not in aspects
        if "battery life is good" in text:
            assert "battery" in aspects or "performance" in aspects or "usability" in aspects
        if "service was a bit slow" in text:
            assert "service" in aspects
        if "the sauce is delicious" in text:
            assert "taste" not in aspects or implicit["extraction_tier"] in {2, 3}

    shutil.rmtree(scratch_root, ignore_errors=True)
    print("dataset_builder regression logic test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
