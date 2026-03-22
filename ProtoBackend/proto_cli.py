from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from implicit_proto.dataset import diagnostics_to_dict, load_dataset_bundle
from implicit_proto.inference import ImplicitAspectDetector
from implicit_proto.test import calibrate_label_thresholds, evaluate_default_backend_split
from implicit_proto.train import train_prototypes


def _progress_iter(iterable, total: int | None = None, desc: str = "", disable: bool = False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("threshold list cannot be empty")
    return values


def _parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("top-k list cannot be empty")
    return values


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": 0}
    return {"path": str(path), "exists": True, "size_bytes": int(path.stat().st_size)}


def _write_artifact_files(
    family_output_dir: Path,
    *,
    dataset_family: str,
    data_source: str,
    model_name: str,
) -> Dict[str, str]:
    artifacts = {
        "prototypes": family_output_dir / "prototypes.npz",
        "label_map": family_output_dir / "label_map.json",
        "encoder_model_dir": family_output_dir / "encoder_model",
        "train_summary": family_output_dir / "train_summary.json",
        "train_data_summary": family_output_dir / "train_data_summary.json",
        "best_config": family_output_dir / "best_config.json",
        "eval_test_best": family_output_dir / "eval_test_best.json",
        "pipeline_report": family_output_dir / "pipeline_report.json",
    }

    manifest = {
        "dataset_family": dataset_family,
        "data_source": data_source,
        "model_name": model_name,
        "artifacts": {name: _file_meta(path) for name, path in artifacts.items()},
    }
    manifest_path = family_output_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)

    backend_payload = {
        "dataset_family": dataset_family,
        "prototypes_path": str(artifacts["prototypes"]),
        "label_map_path": str(artifacts["label_map"]),
        "encoder_model_dir": str(artifacts["encoder_model_dir"]),
        "best_config_path": str(artifacts["best_config"]),
        "eval_test_best_path": str(artifacts["eval_test_best"]),
        "artifact_manifest_path": str(manifest_path),
    }
    backend_artifact_path = family_output_dir / "backend_artifacts.json"
    _write_json(backend_artifact_path, backend_payload)
    return {
        "artifact_manifest": str(manifest_path),
        "backend_artifacts": str(backend_artifact_path),
    }


def _rank_key(row: Dict[str, object], selection_objective: str = "stability_macro") -> tuple[float, float, float, float]:
    train_metrics = row.get("train_label_metrics", {})
    full_metrics = row.get("full_metrics", {})
    diagnostics = row.get("diagnostics", {})
    min_bin_macro = min((diagnostics.get("support_bin_macro_f1", {}) or {"x": 0.0}).values()) if diagnostics else 0.0
    no_pred_rate = float((row.get("summary", {}) or {}).get("no_prediction_rate", 1.0))
    if selection_objective == "stability_macro":
        return (
            float(train_metrics.get("macro_f1", 0.0)),
            float(min_bin_macro),
            -no_pred_rate,
            float(full_metrics.get("micro_f1", 0.0)),
        )
    return (
        float(train_metrics.get("macro_f1", 0.0)),
        float(train_metrics.get("micro_f1", 0.0)),
        float(full_metrics.get("macro_f1", 0.0)),
        -no_pred_rate,
    )


def _run_sweep(
    split: str,
    prototypes_path: Path,
    model_name: str,
    device: str | None,
    dataset_family: str,
    data_source: str,
    input_dir: Path | None,
    thresholds: Sequence[float],
    topks: Sequence[int],
    train_labels: Sequence[str],
    label_thresholds: Mapping[str, float] | None = None,
    show_progress: bool = True,
    selection_objective: str = "stability_macro",
    label_merge_enabled: bool = True,
    label_merge_map: Mapping[str, str] | None = None,
    label_merge_config: str | Path | None = None,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    combos = list(product(topks, thresholds))
    iterator = _progress_iter(combos, total=len(combos), desc=f"Sweep {split}", disable=not show_progress)
    for top_k, threshold in iterator:
        report = evaluate_default_backend_split(
            split=split,
            prototypes_path=prototypes_path,
            top_k=top_k,
            threshold=threshold,
            model_name=model_name,
            device=device,
            dataset_family=dataset_family,
            input_dir=input_dir,
            data_source=data_source,
            show_progress=False,
            train_labels=train_labels,
            label_thresholds=label_thresholds,
            label_merge_enabled=label_merge_enabled,
            label_merge_map=label_merge_map,
            label_merge_config=label_merge_config,
        )
        rows.append(
            {
                "top_k": int(top_k),
                "threshold": float(threshold),
                "summary": report["summary"],
                "full_metrics": report["full_metrics"],
                "train_label_metrics": report["train_label_metrics"],
                "diagnostics": report.get("diagnostics", {}),
            }
        )

    ranked = sorted(rows, key=lambda x: _rank_key(x, selection_objective=selection_objective), reverse=True)
    best = ranked[0] if ranked else None
    return {
        "split": split,
        "model_name": model_name,
        "ranking_metric": selection_objective,
        "results": ranked,
        "best": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ProtoBackend CLI")
    subparsers = parser.add_subparsers(dest="command")

    def add_shared_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset-family", choices=["reviewlevel", "episodic"], default="reviewlevel")
        p.add_argument("--data-source", choices=["backend_raw", "input_dir"], default="backend_raw")
        p.add_argument("--input-dir", type=Path, default=None)
        p.add_argument("--output-dir", type=Path, default=Path("ProtoBackend/outputs"))
        p.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2")
        p.add_argument("--device", type=str, default=None)
        p.add_argument("--no-progress", action="store_true")
        p.add_argument("--allow-degenerate-val", action="store_true")
        p.add_argument("--label-merge-config", type=Path, default=None)
        p.add_argument("--enable-label-merge", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--selection-objective", choices=["stability_macro", "macro_f1"], default="stability_macro")
        p.add_argument("--calibration-min-support", type=int, default=4)

    run_parser = subparsers.add_parser("run", help="Train -> val sweep -> calibrate -> best test eval")
    add_shared_args(run_parser)
    run_parser.add_argument("--batch-size", type=int, default=32)
    run_parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    run_parser.add_argument("--topks", type=str, default="1,2,3,4,5")
    run_parser.add_argument("--dedupe-sentences", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--shrinkage-alpha", type=float, default=4.0)
    run_parser.add_argument("--min-examples-per-centroid", type=int, default=6)
    run_parser.add_argument("--max-centroids-per-label", type=int, default=2)
    run_parser.add_argument("--disable-label-calibration", action="store_true")
    run_parser.add_argument("--low-support-single-centroid-threshold", type=int, default=10)
    run_parser.add_argument("--low-support-shrinkage-boost", type=float, default=2.0)

    train_parser = subparsers.add_parser("train", help="Train prototypes")
    add_shared_args(train_parser)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--train-csv", type=Path, default=None)
    train_parser.add_argument("--dedupe-sentences", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--shrinkage-alpha", type=float, default=4.0)
    train_parser.add_argument("--min-examples-per-centroid", type=int, default=6)
    train_parser.add_argument("--max-centroids-per-label", type=int, default=2)
    train_parser.add_argument("--low-support-single-centroid-threshold", type=int, default=10)
    train_parser.add_argument("--low-support-shrinkage-boost", type=float, default=2.0)

    eval_parser = subparsers.add_parser("eval", help="Evaluate val/test split")
    add_shared_args(eval_parser)
    eval_parser.add_argument("--split", choices=["val", "test"], default="test")
    eval_parser.add_argument("--prototypes", type=Path, default=None)
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--threshold", type=float, default=0.6)
    eval_parser.add_argument("--return-top1-if-empty", action=argparse.BooleanOptionalAction, default=False)

    sweep_parser = subparsers.add_parser("sweep", help="Sweep top-k and threshold")
    add_shared_args(sweep_parser)
    sweep_parser.add_argument("--split", choices=["val", "test"], default="val")
    sweep_parser.add_argument("--prototypes", type=Path, default=None)
    sweep_parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    sweep_parser.add_argument("--topks", type=str, default="1,2,3,4,5")

    predict_parser = subparsers.add_parser("predict", help="Predict aspects for one sentence")
    add_shared_args(predict_parser)
    predict_parser.add_argument("--prototypes", type=Path, default=None)
    predict_parser.add_argument("--sentence", type=str, required=True)
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.add_argument("--threshold", type=float, default=0.6)
    predict_parser.add_argument("--return-top1-if-empty", action=argparse.BooleanOptionalAction, default=True)

    return parser


def _resolve_family_output(output_dir: Path, dataset_family: str) -> Path:
    return _ensure_dir(output_dir / dataset_family)


def _resolve_proto_path(output_dir: Path, dataset_family: str, provided: Path | None) -> Path:
    if provided is not None:
        return provided
    return output_dir / dataset_family / "prototypes.npz"


def _resolve_model_name(family_output_dir: Path, fallback_model_name: str) -> str:
    encoder_dir = family_output_dir / "encoder_model"
    if encoder_dir.exists() and encoder_dir.is_dir():
        return str(encoder_dir)
    return fallback_model_name


def _load_bundle_or_fail(
    dataset_family: str,
    data_source: str,
    input_dir: Path | None,
    allow_degenerate_val: bool,
    label_merge_enabled: bool = True,
    label_merge_config: Path | None = None,
) -> Dict[str, object]:
    bundle = load_dataset_bundle(
        dataset_family=dataset_family,
        data_source=data_source,
        input_dir=input_dir,
        label_merge_enabled=label_merge_enabled,
        label_merge_config=label_merge_config,
    )
    if bundle.diagnostics.is_degenerate_validation and not allow_degenerate_val:
        details = json.dumps(diagnostics_to_dict(bundle.diagnostics), indent=2)
        raise ValueError(
            "Validation split is degenerate for model selection. "
            "Re-run with --allow-degenerate-val to continue anyway.\n"
            f"{details}"
        )
    return {
        "bundle": bundle,
        "diagnostics": diagnostics_to_dict(bundle.diagnostics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    known_commands = {"run", "train", "eval", "sweep", "predict"}
    if not raw_argv or raw_argv[0].startswith("-") or raw_argv[0] not in known_commands:
        raw_argv = ["run", *raw_argv]
    args = parser.parse_args(raw_argv)

    family_output_dir = _resolve_family_output(args.output_dir, args.dataset_family)

    if args.command == "train":
        result = train_prototypes(
            output_dir=family_output_dir,
            train_csv_path=args.train_csv,
            dataset_family=args.dataset_family,
            input_dir=args.input_dir,
            data_source=args.data_source,
            batch_size=args.batch_size,
            model_name=args.model_name,
            device=args.device,
            dedupe_sentences=args.dedupe_sentences,
            shrinkage_alpha=args.shrinkage_alpha,
            min_examples_per_centroid=args.min_examples_per_centroid,
            max_centroids_per_label=args.max_centroids_per_label,
            low_support_single_centroid_threshold=args.low_support_single_centroid_threshold,
            low_support_shrinkage_boost=args.low_support_shrinkage_boost,
            fail_on_degenerate_val=not args.allow_degenerate_val,
            label_merge_enabled=args.enable_label_merge,
            label_merge_config=args.label_merge_config,
        )
        result.update(
            _write_artifact_files(
                family_output_dir,
                dataset_family=args.dataset_family,
                data_source=args.data_source,
                model_name=args.model_name,
            )
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "eval":
        bundle_meta = _load_bundle_or_fail(
            dataset_family=args.dataset_family,
            data_source=args.data_source,
            input_dir=args.input_dir,
            allow_degenerate_val=args.allow_degenerate_val or args.split != "val",
            label_merge_enabled=args.enable_label_merge,
            label_merge_config=args.label_merge_config,
        )
        bundle = bundle_meta["bundle"]
        proto_path = _resolve_proto_path(args.output_dir, args.dataset_family, args.prototypes)
        report = evaluate_default_backend_split(
            split=args.split,
            prototypes_path=proto_path,
            top_k=args.top_k,
            threshold=args.threshold,
            model_name=_resolve_model_name(family_output_dir, args.model_name),
            device=args.device,
            dataset_family=args.dataset_family,
            input_dir=args.input_dir,
            data_source=args.data_source,
            show_progress=not args.no_progress,
            return_top1_if_empty=args.return_top1_if_empty,
            train_labels=bundle.diagnostics.train_labels,
            label_merge_enabled=args.enable_label_merge,
            label_merge_config=args.label_merge_config,
        )
        out_path = family_output_dir / f"eval_{args.split}.json"
        _write_json(out_path, report)
        _write_artifact_files(
            family_output_dir,
            dataset_family=args.dataset_family,
            data_source=args.data_source,
            model_name=args.model_name,
        )
        print(json.dumps(report["summary"], indent=2))
        print(f"Saved: {out_path}")
        return 0

    if args.command == "sweep":
        bundle_meta = _load_bundle_or_fail(
            dataset_family=args.dataset_family,
            data_source=args.data_source,
            input_dir=args.input_dir,
            allow_degenerate_val=args.allow_degenerate_val or args.split != "val",
            label_merge_enabled=args.enable_label_merge,
            label_merge_config=args.label_merge_config,
        )
        bundle = bundle_meta["bundle"]
        proto_path = _resolve_proto_path(args.output_dir, args.dataset_family, args.prototypes)
        thresholds = _parse_float_list(args.thresholds)
        topks = _parse_int_list(args.topks)
        payload = _run_sweep(
            split=args.split,
            prototypes_path=proto_path,
            model_name=_resolve_model_name(family_output_dir, args.model_name),
            device=args.device,
            dataset_family=args.dataset_family,
            data_source=args.data_source,
            input_dir=args.input_dir,
            thresholds=thresholds,
            topks=topks,
            train_labels=bundle.diagnostics.train_labels,
            selection_objective=args.selection_objective,
            label_merge_enabled=args.enable_label_merge,
            label_merge_config=args.label_merge_config,
            show_progress=not args.no_progress,
        )
        timestamp = int(time.time())
        out_json = family_output_dir / f"sweep_results_{args.split}_{timestamp}.json"
        best_path = family_output_dir / "best_config.json"
        _write_json(out_json, payload)
        _write_json(best_path, payload.get("best") or {})
        _write_artifact_files(
            family_output_dir,
            dataset_family=args.dataset_family,
            data_source=args.data_source,
            model_name=args.model_name,
        )
        print(json.dumps(payload["best"], indent=2))
        print(f"Saved: {out_json}")
        print(f"Saved: {best_path}")
        return 0

    if args.command == "predict":
        proto_path = _resolve_proto_path(args.output_dir, args.dataset_family, args.prototypes)
        detector = ImplicitAspectDetector.from_artifacts(
            prototypes_path=proto_path,
            model_name=_resolve_model_name(family_output_dir, args.model_name),
            device=args.device,
        )
        out = detector.predict_aspect_dicts(
            sentence=args.sentence,
            top_k=args.top_k,
            threshold=args.threshold,
            return_top1_if_empty=args.return_top1_if_empty,
        )
        print(json.dumps(out, indent=2))
        return 0

    bundle_meta = _load_bundle_or_fail(
        dataset_family=args.dataset_family,
        data_source=args.data_source,
        input_dir=args.input_dir,
        allow_degenerate_val=args.allow_degenerate_val,
        label_merge_enabled=args.enable_label_merge,
        label_merge_config=args.label_merge_config,
    )
    bundle = bundle_meta["bundle"]
    thresholds = _parse_float_list(args.thresholds)
    topks = _parse_int_list(args.topks)

    print("[1/4] Training prototypes...")
    train_result = train_prototypes(
        output_dir=family_output_dir,
        dataset_family=args.dataset_family,
        data_source=args.data_source,
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        model_name=args.model_name,
        device=args.device,
        dedupe_sentences=args.dedupe_sentences,
        shrinkage_alpha=args.shrinkage_alpha,
        min_examples_per_centroid=args.min_examples_per_centroid,
        max_centroids_per_label=args.max_centroids_per_label,
        low_support_single_centroid_threshold=args.low_support_single_centroid_threshold,
        low_support_shrinkage_boost=args.low_support_shrinkage_boost,
        fail_on_degenerate_val=not args.allow_degenerate_val,
        label_merge_enabled=args.enable_label_merge,
        label_merge_config=args.label_merge_config,
    )
    proto_path = Path(str(train_result["prototypes"]))

    print("[2/4] Sweeping on validation split...")
    val_sweep = _run_sweep(
        split="val",
        prototypes_path=proto_path,
        model_name=_resolve_model_name(family_output_dir, args.model_name),
        device=args.device,
        dataset_family=args.dataset_family,
        data_source=args.data_source,
        input_dir=args.input_dir,
        thresholds=thresholds,
        topks=topks,
        train_labels=bundle.diagnostics.train_labels,
        selection_objective=args.selection_objective,
        label_merge_enabled=args.enable_label_merge,
        label_merge_config=args.label_merge_config,
        show_progress=not args.no_progress,
    )
    timestamp = int(time.time())
    val_sweep_path = family_output_dir / f"sweep_results_val_{timestamp}.json"
    _write_json(val_sweep_path, val_sweep)
    best = val_sweep.get("best") or {}

    best_top_k = int(best.get("top_k", 3))
    best_threshold = float(best.get("threshold", 0.6))

    print("[3/4] Calibrating per-label thresholds on validation split...")
    detector = ImplicitAspectDetector.from_artifacts(
        prototypes_path=proto_path,
        model_name=_resolve_model_name(family_output_dir, args.model_name),
        device=args.device,
    )
    label_thresholds = {}
    if not args.disable_label_calibration:
        label_thresholds = calibrate_label_thresholds(
            detector=detector,
            dataset=bundle.splits["val"],
            candidate_thresholds=thresholds,
            train_labels=bundle.diagnostics.train_labels,
            top_k=best_top_k,
            base_threshold=best_threshold,
            min_support=max(1, int(args.calibration_min_support)),
            min_threshold=min(thresholds),
            max_threshold=max(thresholds),
        )

    best_config = {
        **best,
        "label_thresholds": label_thresholds,
    }
    best_cfg_path = family_output_dir / "best_config.json"
    _write_json(best_cfg_path, best_config)

    print("[4/4] Evaluating best config on test split...")
    test_report = evaluate_default_backend_split(
        split="test",
        prototypes_path=proto_path,
        top_k=best_top_k,
        threshold=best_threshold,
        model_name=_resolve_model_name(family_output_dir, args.model_name),
        device=args.device,
        dataset_family=args.dataset_family,
        input_dir=args.input_dir,
        data_source=args.data_source,
        show_progress=not args.no_progress,
        train_labels=bundle.diagnostics.train_labels,
        label_thresholds=label_thresholds,
        label_merge_enabled=args.enable_label_merge,
        label_merge_config=args.label_merge_config,
    )
    test_path = family_output_dir / "eval_test_best.json"
    _write_json(test_path, test_report)

    pipeline_report = {
        "dataset_family": args.dataset_family,
        "data_source": args.data_source,
        "model_name": _resolve_model_name(family_output_dir, args.model_name),
        "input_dir": str(args.input_dir) if args.input_dir else None,
        "split_paths": bundle.split_paths,
        "dataset_diagnostics": diagnostics_to_dict(bundle.diagnostics),
        "prototypes_path": str(proto_path),
        "train": train_result,
        "val_sweep_path": str(val_sweep_path),
        "best_config": best_config,
        "test_summary": test_report["summary"],
        "test_report_path": str(test_path),
    }
    pipeline_path = family_output_dir / "pipeline_report.json"
    _write_json(pipeline_path, pipeline_report)
    artifact_paths = _write_artifact_files(
        family_output_dir,
        dataset_family=args.dataset_family,
        data_source=args.data_source,
        model_name=args.model_name,
    )

    print(json.dumps(best_config, indent=2))
    print(json.dumps(test_report["summary"], indent=2))
    print(f"Saved: {pipeline_path}")
    print(f"Saved: {artifact_paths['artifact_manifest']}")
    print(f"Saved: {artifact_paths['backend_artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
