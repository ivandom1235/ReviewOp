from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from dataset_builder.config import (
    BuilderConfig, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, SUPPORTED_LLM_PROVIDERS, 
    validate_config, get_default_llm_model, get_default_llm_provider
)

from dataset_builder.canonical.canonicalizer import canonicalize_label
from dataset_builder.ingest.loaders import load_csv_reviews, load_jsonl_reviews
from dataset_builder.schemas.raw_review import RawReview
from dataset_builder.schemas.benchmark_row import BenchmarkRow
from dataset_builder.schemas.interpretation import Interpretation
from dataset_builder.orchestrator.pipeline import run_builder_pipeline
from dataset_builder.split.grouped_split import grouped_train_val_test_split
from dataset_builder.verify.openai_verifier import OpenAIVerifier
from rich.progress import track, Progress


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm", choices=sorted(SUPPORTED_LLM_PROVIDERS), default=get_default_llm_provider())
    parser.add_argument("--llm-model", default=get_default_llm_model())
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--symptom-store", type=Path, default=None, help="Path to learned symptom patterns JSON")
    return parser


def resolve_input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise ValueError(f"input path is neither file nor directory: {path}")
    candidates = sorted([*path.glob("*.jsonl"), *path.glob("*.csv")])
    if not candidates:
        raise FileNotFoundError(f"no .jsonl or .csv files found in {path}")
    return candidates


def resolve_input_path(path: Path) -> Path:
    return resolve_input_paths(path)[0]


def build_config_from_args(args: argparse.Namespace, resolved_input_path: Path) -> BuilderConfig:
    from dataset_builder.config import get_env_model
    input_path = Path(args.input)
    
    # Use args.llm_model if provided, otherwise check env
    llm_model = args.llm_model
    if llm_model == get_default_llm_model():
        llm_model = get_env_model(args.llm, llm_model)

    cfg = BuilderConfig(
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        input_paths=(resolved_input_path,),
        output_dir=args.output_dir,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        llm_provider=args.llm,
        llm_model=llm_model,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        symptom_store_path=str(args.symptom_store) if args.symptom_store else None,
    )
    validate_config(cfg)
    return cfg


def select_working_reviews(rows: Sequence[RawReview], cfg: BuilderConfig) -> list[RawReview]:
    ordered = list(rows)
    random.Random(cfg.random_seed).shuffle(ordered)
    if cfg.sample_size is not None:
        ordered = ordered[: cfg.sample_size]
    if cfg.chunk_size is not None:
        start = cfg.chunk_offset
        ordered = ordered[start : start + cfg.chunk_size]
    return ordered


def load_reviews(paths: Sequence[Path]) -> list[RawReview]:
    rows: list[RawReview] = []
    for path in track(paths, description="Loading reviews..."):
        rows.extend(load_jsonl_reviews(path) if path.suffix.lower() == ".jsonl" else load_csv_reviews(path))
    return rows




def main() -> None:
    from dataset_builder.profile.dataset_profiler import profile_dataset
    args = build_arg_parser().parse_args()
    paths = resolve_input_paths(args.input)
    rows = load_reviews(paths)
    cfg = build_config_from_args(args, paths[0])
    cfg = replace(cfg, input_paths=tuple(paths))
    rows = select_working_reviews(rows, cfg)
    
    profile = profile_dataset(rows)
    run_builder_pipeline(cfg, raw_reviews=rows, profile_summary=profile)


if __name__ == "__main__":
    main()
