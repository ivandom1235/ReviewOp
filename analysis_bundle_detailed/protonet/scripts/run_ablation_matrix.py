from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from protonet.config import ECProtoNetV2Config
from protonet.io_utils import write_json
from protonet.pipeline import run_compare


def main() -> None:
    p = argparse.ArgumentParser(description="Run ablation matrix for grouped/domain-holdout protocols.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--grouped-config", default=None, help="Optional JSON config path for grouped baseline profile.")
    p.add_argument("--domain-config", default=None, help="Optional JSON config path for domain-holdout baseline profile.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir

    grouped_base = (
        ECProtoNetV2Config.from_json(Path(args.grouped_config))
        if args.grouped_config
        else ECProtoNetV2Config(require_artifact_pass=False, require_active_contract=False)
    )
    domain_base = (
        ECProtoNetV2Config.from_json(Path(args.domain_config))
        if args.domain_config
        else grouped_base
    )
    variants = {
        "default": {},
        "no_memory": {"use_memory": False},
        "review_queue_only": {"use_memory": True, "memory_source_mode": "review_queue", "memory_min_status": "detected"},
        "promoted_memory": {"use_memory": True, "memory_source_mode": "promoted", "memory_min_status": "promoted"},
        "illegal_oracle_memory": {"use_memory": True, "memory_source_mode": "oracle_all", "memory_min_status": "detected"},
        "no_open_world": {"emit_open_world_candidate": False},
        "no_equivalence_mapping": {"use_equivalence": False},
        "no_evidence_prototypes": {"use_evidence_text_for_prototypes": False},
        "no_aspect_name_support": {"include_aspect_name_in_support_text": False},
        "no_boundary_routing": {"boundary_margin_threshold": 0.0},
        "residual_energy_confidence": {
            "open_world_unknown_residual_weight": 0.45,
            "open_world_unknown_energy_weight": 0.35,
            "open_world_unknown_confidence_weight": 0.20,
        },
        "energy_only": {
            "open_world_unknown_residual_weight": 0.0,
            "open_world_unknown_energy_weight": 1.0,
            "open_world_unknown_confidence_weight": 0.0,
        },
        "residual_only": {
            "open_world_unknown_residual_weight": 1.0,
            "open_world_unknown_energy_weight": 0.0,
            "open_world_unknown_confidence_weight": 0.0,
        },
        "residual_energy": {
            "open_world_unknown_residual_weight": 0.5,
            "open_world_unknown_energy_weight": 0.5,
            "open_world_unknown_confidence_weight": 0.0,
        },
        "evidence_low": {"w_evidence": 0.10},
        "evidence_high": {"w_evidence": 0.35},
    }

    results: dict[str, dict] = {}
    for name, overrides in variants.items():
        grouped_cfg = replace(grouped_base, **overrides)
        domain_cfg = replace(domain_base, **overrides)
        grouped_dir = out / name / "grouped"
        domain_dir = out / name / "domain_holdout"
        grouped = run_compare(
            artifact_dir=artifact_dir,
            output_dir=grouped_dir,
            config=grouped_cfg,
            protocol="grouped",
            split="test",
            allow_failed_artifact=True,
        )["metrics"]
        domain = run_compare(
            artifact_dir=artifact_dir,
            output_dir=domain_dir,
            config=domain_cfg,
            protocol="domain_holdout",
            split="test",
            allow_failed_artifact=True,
        )["metrics"]
        results[name] = {
            "grouped": {
                "final_strict_f1": grouped.get("final_strict_f1"),
                "coverage": grouped.get("coverage"),
                "unseen_f1": grouped.get("unseen_detection", {}).get("f1"),
                "unknown_auroc": grouped.get("unknown_auroc"),
                "unknown_auprc": grouped.get("unknown_auprc"),
                "memory": grouped.get("memory", {}),
            },
            "domain_holdout": {
                "final_strict_f1": domain.get("final_strict_f1"),
                "coverage": domain.get("coverage"),
                "unseen_f1": domain.get("unseen_detection", {}).get("f1"),
                "unknown_auroc": domain.get("unknown_auroc"),
                "unknown_auprc": domain.get("unknown_auprc"),
                "memory": domain.get("memory", {}),
            },
        }

    write_json(out / "ablation_summary.json", results)
    print(results)


if __name__ == "__main__":
    main()
