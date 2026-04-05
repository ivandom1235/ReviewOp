from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .config import METADATA_ROOT, OUTPUT_ROOT, ProtonetConfig, resolve_default_input_dir, seed_everything
    from .dataset_reader import load_input_dataset, write_jsonl
    from .episode_builder import build_or_load_episode_sets
    from .evaluator import evaluate_episodes
    from .export_bundle import export_model_bundle, export_report
    from .model import ProtoNetModel
    from .reviewlevel_adapter import adapt_reviewlevel_rows
    from .trainer import load_checkpoint, train_model
except ImportError:
    from config import METADATA_ROOT, OUTPUT_ROOT, ProtonetConfig, resolve_default_input_dir, seed_everything
    from dataset_reader import load_input_dataset, write_jsonl
    from episode_builder import build_or_load_episode_sets
    from evaluator import evaluate_episodes
    from export_bundle import export_model_bundle, export_report
    from model import ProtoNetModel
    from reviewlevel_adapter import adapt_reviewlevel_rows
    from trainer import load_checkpoint, train_model


def _build_config(args: argparse.Namespace) -> ProtonetConfig:
    input_root = Path(args.input_dir) if args.input_dir else resolve_default_input_dir(args.input_type)
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else METADATA_ROOT
    return ProtonetConfig(
        input_type=args.input_type,
        input_dir=input_root,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        checkpoint_dir=output_dir / "checkpoints",
        episode_cache_dir=output_dir / "episodes",
        predictions_dir=output_dir / "predictions",
        encoder_backend=args.encoder_backend,
        encoder_model_name=args.encoder_model_name,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        max_train_episodes=args.max_train_episodes,
        max_eval_episodes=args.max_eval_episodes,
        contrastive_weight=args.contrastive_weight,
        prototype_smoothing=args.prototype_smoothing,
        low_confidence_threshold=args.low_confidence_threshold,
        seed=args.seed,
        no_progress=args.no_progress,
        force_rebuild_episodes=args.force_rebuild_episodes,
        strict_encoder=args.strict_encoder,
        production_require_transformer=args.production_require_transformer,
        allow_model_download=args.allow_model_download,
        compile_model=args.compile_model,
    )


def _prepare_examples(cfg: ProtonetConfig):
    rows_by_split, summary = load_input_dataset(cfg)
    if cfg.input_type == "reviewlevel":
        rows_by_split = adapt_reviewlevel_rows(rows_by_split, progress_enabled=cfg.progress_enabled)
    return rows_by_split, summary


def run_train(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    cfg.ensure_dirs()
    seed_everything(cfg.seed)
    rows_by_split, summary = _prepare_examples(cfg)
    episodes_by_split = build_or_load_episode_sets(rows_by_split, cfg)
    result = train_model(cfg, episodes_by_split)

    if cfg.save_predictions:
        val_metrics, val_predictions = evaluate_episodes(result.model, episodes_by_split["val"], cfg, "val")
        test_metrics, test_predictions = evaluate_episodes(result.model, episodes_by_split["test"], cfg, "test")
        write_jsonl(cfg.predictions_dir / "val_predictions.jsonl", val_predictions)
        write_jsonl(cfg.predictions_dir / "test_predictions.jsonl", test_predictions)
    else:
        val_metrics = result.val_metrics
        test_metrics = result.test_metrics

    bundle_path = export_model_bundle(
        cfg=cfg,
        model=result.model,
        prototype_bank=result.prototype_bank,
        checkpoint_path=result.checkpoint_path,
        metrics={"val": val_metrics, "test": test_metrics},
        history=result.history,
    )
    report_path = export_report(
        cfg=cfg,
        input_summary={
            "split_sizes": summary.split_sizes,
            "detected_format": summary.detected_format,
            "input_type": summary.input_type,
            "episode_counts": {split: len(rows) for split, rows in episodes_by_split.items()},
        },
        train_metrics=result.history[-1] if result.history else {},
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        bundle_path=bundle_path,
        checkpoint_path=result.checkpoint_path,
        history=result.history,
    )
    print(f"Training complete. Report: {report_path}")
    return 0


def run_eval(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    cfg.ensure_dirs()
    seed_everything(cfg.seed)
    rows_by_split, _ = _prepare_examples(cfg)
    episodes_by_split = build_or_load_episode_sets(rows_by_split, cfg)
    model = ProtoNetModel(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "best.pt"
    load_checkpoint(model, checkpoint_path)
    split = args.split
    metrics, predictions = evaluate_episodes(model, episodes_by_split[split], cfg, split)
    write_jsonl(cfg.predictions_dir / f"{split}_predictions.jsonl", predictions)
    print(metrics)
    return 0


def run_export(args: argparse.Namespace) -> int:
    cfg = _build_config(args)
    cfg.ensure_dirs()
    seed_everything(cfg.seed)
    rows_by_split, _ = _prepare_examples(cfg)
    episodes_by_split = build_or_load_episode_sets(rows_by_split, cfg)
    model = ProtoNetModel(cfg)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "best.pt"
    load_checkpoint(model, checkpoint_path)
    try:
        from .prototype_bank import build_global_prototype_bank
    except ImportError:
        from prototype_bank import build_global_prototype_bank

    bank = build_global_prototype_bank(model, episodes_by_split["train"], cfg)
    bundle_path = export_model_bundle(
        cfg=cfg,
        model=model,
        prototype_bank=bank,
        checkpoint_path=checkpoint_path,
        metrics={},
        history=[],
    )
    print(f"Exported bundle: {bundle_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ProtoNet training pipeline")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-type", choices=["episodic", "reviewlevel"], default="episodic")
    common.add_argument("--input-dir", type=str, default=None)
    common.add_argument("--output-dir", type=str, default=None)
    common.add_argument("--metadata-dir", type=str, default=None)
    common.add_argument("--encoder-backend", choices=["auto", "transformer", "bow"], default="auto")
    common.add_argument("--encoder-model-name", type=str, default="microsoft/deberta-v3-base")
    common.add_argument("--n-way", type=int, default=3)
    common.add_argument("--k-shot", type=int, default=2)
    common.add_argument("--q-query", type=int, default=2)
    common.add_argument("--epochs", type=int, default=8)
    common.add_argument("--warmup-epochs", type=int, default=1)
    common.add_argument("--patience", type=int, default=3)
    common.add_argument("--max-train-episodes", type=int, default=120)
    common.add_argument("--max-eval-episodes", type=int, default=48)
    common.add_argument("--contrastive-weight", type=float, default=0.15)
    common.add_argument("--prototype-smoothing", type=float, default=0.05)
    common.add_argument("--low-confidence-threshold", type=float, default=0.55)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--no-progress", action="store_true")
    common.add_argument("--force-rebuild-episodes", action="store_true")
    common.add_argument("--strict-encoder", action="store_true")
    common.add_argument("--production-require-transformer", action="store_true")
    common.add_argument("--allow-model-download", action="store_true")
    common.add_argument("--compile-model", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train, validate, test, and export", parents=[common])

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained checkpoint", parents=[common])
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--checkpoint", type=str, default=None)

    export_parser = subparsers.add_parser("export", help="Export a portable bundle from a checkpoint", parents=[common])
    export_parser.add_argument("--checkpoint", type=str, default=None)
    return parser


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        return None
    commands = {"train", "eval", "export"}
    for index, token in enumerate(argv):
        if token in commands:
            if index == 0:
                return argv
            return [token, *argv[:index], *argv[index + 1 :]]
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    if args.command == "train":
        return run_train(args)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "export":
        return run_export(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
