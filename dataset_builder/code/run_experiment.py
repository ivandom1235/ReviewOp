from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from build_dataset import run_pipeline
from contracts import BuilderConfig
from experiments import run_experiments
from research_stack import build_experiment_plan, benchmark_registry_payload, model_registry_payload
from utils import stable_id, utc_now_iso, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset builder research experiment runner")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-offset", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--max-aspects", type=int, default=20)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--implicit-mode", type=str, default="heuristic", choices=["heuristic", "benchmark"])
    parser.add_argument("--benchmark-key", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="heuristic_latent")
    parser.add_argument("--augmentation-mode", type=str, default="none")
    parser.add_argument("--prompt-mode", type=str, default="constrained")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-baseline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = BuilderConfig(
        input_dir=args.input_dir or BuilderConfig().input_dir,
        output_dir=args.output_dir or BuilderConfig().output_dir,
        random_seed=args.seed,
        text_column_override=args.text_column,
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_offset=args.chunk_offset,
        confidence_threshold=args.confidence_threshold,
        max_aspects=args.max_aspects,
        min_text_tokens=args.min_text_tokens,
        implicit_mode=args.implicit_mode,
        benchmark_key=args.benchmark_key,
        model_family=args.model_family,
        augmentation_mode=args.augmentation_mode,
        prompt_mode=args.prompt_mode,
    )

    run_id = stable_id(
        cfg.input_dir,
        cfg.output_dir,
        cfg.benchmark_key or "auto",
        cfg.model_family,
        cfg.random_seed,
        cfg.sample_size,
        cfg.chunk_size,
        cfg.chunk_offset,
        cfg.implicit_mode,
    )
    run_dir = cfg.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "benchmark_registry.json", benchmark_registry_payload())
    write_json(run_dir / "model_registry.json", model_registry_payload())
    write_json(run_dir / "experiment_plan.json", [asdict(item) for item in build_experiment_plan()])
    write_json(run_dir / "base_config.json", asdict(cfg))

    if args.plan_only:
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "planned",
            "config": asdict(cfg),
        })
        return 0

    if args.execute_baseline:
        baseline_cfg = replace(cfg)
        report = run_pipeline(baseline_cfg)
        write_json(run_dir / "baseline_report.json", report)
        write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "status": "completed",
            "config": asdict(cfg),
            "report": report,
        })
        return 0

    run_experiments(cfg, [{"model_family": cfg.model_family, "benchmark_key": cfg.benchmark_key}], run_dir)
    write_json(run_dir / "manifest.json", {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "status": "configured",
        "config": asdict(cfg),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
