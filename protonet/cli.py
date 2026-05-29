from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .artifact_guard import assert_verified_artifact
from .calibration import grid_search_router
from .config import ECProtoNetV2Config
from .dataset import load_dataset_bundle
from .io_utils import write_json
from .pipeline import run_compare
from .reproducibility import seed_everything


def build_config(args: argparse.Namespace) -> ECProtoNetV2Config:
    cfg = ECProtoNetV2Config()
    if args.config:
        path = Path(args.config)
        cfg = ECProtoNetV2Config.from_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else ECProtoNetV2Config.from_json(path)
    updates = {}
    for key in [
        "encoder",
        "model_name",
        "top_k",
        "accept_threshold",
        "abstain_threshold",
        "novel_threshold",
        "boundary_margin_threshold",
        "evidence_abstain_threshold",
    ]:
        if hasattr(args, key) and getattr(args, key) is not None:
            updates[key] = getattr(args, key)
    if getattr(args, "allow_missing_active_contract", False):
        updates["require_active_contract"] = False
    if getattr(args, "allow_failed_artifact", False):
        updates["require_artifact_pass"] = False
        updates["require_active_contract"] = False
    if getattr(args, "use_memory", None) is not None:
        updates["use_memory"] = args.use_memory
    return replace(cfg, **updates)


def cmd_verify(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    report = assert_verified_artifact(
        args.artifact_dir,
        cfg,
        allow_stale_source_artifact=getattr(args, "allow_stale_source_artifact", False),
    )
    print(json.dumps(report.to_dict(), indent=2))


def cmd_summary(args: argparse.Namespace) -> None:
    bundle = load_dataset_bundle(args.artifact_dir)
    print(json.dumps(bundle.summary(), indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    result = run_compare(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        config=cfg,
        split=args.split,
        protocol=args.protocol,
        allow_failed_artifact=args.allow_failed_artifact,
        allow_stale_source_artifact=getattr(args, "allow_stale_source_artifact", False),
    )
    print(json.dumps(result["metrics"], indent=2))


def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    # Calibration can inspect val before active artifact contract exists, but artifact must still pass unless override supplied.
    assert_verified_artifact(
        args.artifact_dir,
        cfg,
        allow_stale_source_artifact=getattr(args, "allow_stale_source_artifact", False),
    )
    bundle = load_dataset_bundle(args.artifact_dir)
    result = grid_search_router(bundle, replace(cfg, require_active_contract=False))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "calibration_results.json", result)
    if result.get("best"):
        write_json(out / "best_router_config.json", result["best"]["config"])
    print(json.dumps(result["best"], indent=2))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EC-ProtoNet V2 CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", type=str, default=None)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--encoder", type=str, default=None, choices=["hashing", "sentence-transformers"])
        sp.add_argument("--model-name", type=str, default=None)
        sp.add_argument("--top-k", type=int, default=None)
        sp.add_argument("--accept-threshold", type=float, default=None)
        sp.add_argument("--abstain-threshold", type=float, default=None)
        sp.add_argument("--novel-threshold", type=float, default=None)
        sp.add_argument("--boundary-margin-threshold", type=float, default=None)
        sp.add_argument("--evidence-abstain-threshold", type=float, default=None)
        sp.add_argument("--allow-missing-active-contract", action="store_true")
        sp.add_argument("--allow-failed-artifact", action="store_true")
        sp.add_argument("--allow-stale-source-artifact", action="store_true")
        sp.add_argument("--use-memory", dest="use_memory", action="store_true", default=None)
        sp.add_argument("--disable-memory", dest="use_memory", action="store_false")

    sp = sub.add_parser("verify-artifact")
    sp.add_argument("--artifact-dir", required=True)
    common(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("summary")
    sp.add_argument("--artifact-dir", required=True)
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("compare")
    sp.add_argument("--artifact-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--protocol", default="grouped", choices=["grouped", "domain_holdout"])
    sp.add_argument("--split", default="test", choices=["train", "val", "test"])
    common(sp)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("calibrate")
    sp.add_argument("--artifact-dir", required=True)
    sp.add_argument("--output-dir", required=True)
    common(sp)
    sp.set_defaults(func=cmd_calibrate)

    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    seed_everything(getattr(args, "seed", 42))
    args.func(args)


if __name__ == "__main__":
    main()
