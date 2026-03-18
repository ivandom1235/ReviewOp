from __future__ import annotations

import argparse
import sys

from build_dataset import main as build_main


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility entrypoint for dataset builder")
    parser.add_argument("--input", dest="input_dir", default="dataset_builder/input")
    parser.add_argument("--output", dest="output_dir", default="dataset_builder/output")
    parser.add_argument("--use-openai", dest="use_openai", default="false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-ratios", default="0.8,0.1,0.1")
    parser.add_argument("--max-aspects", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--prefer-open-aspect", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # OpenAI toggle is accepted for forward compatibility; current pipeline is rule-based.
    _ = str_to_bool(args.use_openai)

    sys.argv = [
        "build_dataset.py",
        "--input-dir",
        args.input_dir,
        "--output-dir",
        args.output_dir,
        "--split-ratios",
        args.split_ratios,
        "--max-aspects",
        str(args.max_aspects),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--seed",
        str(args.seed),
    ]

    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.prefer_open_aspect:
        sys.argv.append("--prefer-open-aspect")

    build_main()


if __name__ == "__main__":
    main()
