from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from protonet.config import ECProtoNetV2Config
from protonet.io_utils import read_json, write_json
from protonet.pipeline import run_compare


def main() -> None:
    p = argparse.ArgumentParser(description="Run grouped and domain-holdout evaluations in one manifested run.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--config", default=None)
    p.add_argument("--allow-failed-artifact", action="store_true")
    args = p.parse_args()

    root = Path(args.output_dir)
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / run_id
    grouped_dir = run_dir / "grouped" / args.split
    domain_dir = run_dir / "domain_holdout" / args.split

    if args.config:
        path = Path(args.config)
        cfg = ECProtoNetV2Config.from_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else ECProtoNetV2Config.from_json(path)
    else:
        cfg = ECProtoNetV2Config()

    if args.allow_failed_artifact:
        cfg = replace(cfg, require_artifact_pass=False, require_active_contract=False)

    run_compare(
        artifact_dir=args.artifact_dir,
        output_dir=grouped_dir,
        config=cfg,
        protocol="grouped",
        split=args.split,
        allow_failed_artifact=args.allow_failed_artifact,
    )
    run_compare(
        artifact_dir=args.artifact_dir,
        output_dir=domain_dir,
        config=cfg,
        protocol="domain_holdout",
        split=args.split,
        allow_failed_artifact=args.allow_failed_artifact,
    )

    grouped_meta = read_json(grouped_dir / "run_metadata.json", {})
    domain_meta = read_json(domain_dir / "run_metadata.json", {})
    grouped_metrics = read_json(grouped_dir / "metrics.json", {})
    domain_metrics = read_json(domain_dir / "metrics.json", {})

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "split": args.split,
        "code_hash": grouped_meta.get("code_hash"),
        "config_hash": grouped_meta.get("config_hash"),
        "hash_consistent": grouped_meta.get("code_hash") == domain_meta.get("code_hash")
        and grouped_meta.get("config_hash") == domain_meta.get("config_hash"),
        "grouped": {
            "output_dir": str(grouped_dir),
            "final_strict_f1": grouped_metrics.get("final_strict_f1"),
            "known_recall": grouped_metrics.get("known_inventory", {}).get("strict", {}).get("recall"),
        },
        "domain_holdout": {
            "output_dir": str(domain_dir),
            "final_strict_f1": domain_metrics.get("final_strict_f1"),
            "known_recall": domain_metrics.get("known_inventory", {}).get("strict", {}).get("recall"),
            "named_unseen_f1": domain_metrics.get("named_open_world", {}).get("named_unseen_f1", {}).get("f1"),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
