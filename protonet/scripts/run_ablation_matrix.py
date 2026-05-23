from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from protonet.config import ECProtoNetV2Config
from protonet.io_utils import read_json, read_jsonl, write_json
from protonet.pipeline import run_compare


def _prediction_map(pred_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not pred_path.exists():
        return out
    for row in read_jsonl(pred_path):
        out[str(row.get("row_id"))] = row
    return out


def _memory_effect_report(no_mem_dir: Path, promoted_dir: Path) -> dict:
    no_mem = _prediction_map(no_mem_dir / "predictions.jsonl")
    promoted = _prediction_map(promoted_dir / "predictions.jsonl")
    changed_pred = 0
    changed_top1_decision = 0
    changed_unknown = 0
    changed_rows: list[dict] = []
    row_ids = sorted(set(no_mem.keys()) & set(promoted.keys()))
    for rid in row_ids:
        a = no_mem[rid]
        b = promoted[rid]
        a_pred = tuple(a.get("predicted_labels") or [])
        b_pred = tuple(b.get("predicted_labels") or [])
        if a_pred != b_pred:
            changed_pred += 1
        a_cands = a.get("candidates") or []
        b_cands = b.get("candidates") or []
        a_top = a_cands[0] if a_cands else {}
        b_top = b_cands[0] if b_cands else {}
        if str(a_top.get("decision")) != str(b_top.get("decision")):
            changed_top1_decision += 1
        if float(a_top.get("unknown_score") or 0.0) != float(b_top.get("unknown_score") or 0.0):
            changed_unknown += 1
        if (
            a_pred != b_pred
            or str(a_top.get("decision")) != str(b_top.get("decision"))
            or float(a_top.get("unknown_score") or 0.0) != float(b_top.get("unknown_score") or 0.0)
        ):
            changed_rows.append(
                {
                    "row_id": rid,
                    "no_memory_predicted_labels": list(a_pred),
                    "promoted_memory_predicted_labels": list(b_pred),
                    "no_memory_top1_decision": a_top.get("decision"),
                    "promoted_memory_top1_decision": b_top.get("decision"),
                    "no_memory_top1_unknown_score": float(a_top.get("unknown_score") or 0.0),
                    "promoted_memory_top1_unknown_score": float(b_top.get("unknown_score") or 0.0),
                }
            )
    promoted_metrics = {}
    metrics_path = promoted_dir / "metrics.json"
    if metrics_path.exists():
        try:
            promoted_metrics = read_json(metrics_path, default={}) or {}
        except Exception:
            promoted_metrics = {}

    memory_entry_count = 0
    promoted_matches_used = 0
    if isinstance(promoted_metrics, dict):
        memory_section = promoted_metrics.get("memory", {})
        if isinstance(memory_section, dict):
            memory_entry_count = int(memory_section.get("promoted_entries_total", 0) or 0)
            promoted_matches_used = int(memory_section.get("promoted_matches_used", memory_section.get("matches_used", 0)) or 0)

    return {
        "memory_entry_count": int(memory_entry_count),
        "promoted_matches_used": int(promoted_matches_used),
        "changed_prediction_count": int(changed_pred),
        "changed_route_count": int(changed_top1_decision),
        "rows": changed_rows[:200],
        # Backward-compatible aliases
        "rows_compared": len(row_ids),
        "memory_changed_prediction_count": int(changed_pred),
        "memory_changed_top1_decision_count": int(changed_top1_decision),
        "memory_changed_top1_unknown_score_count": int(changed_unknown),
        "changed_rows": changed_rows[:200],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run ablation matrix for grouped/domain-holdout protocols.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--encoder", default="hashing", choices=["hashing", "sentence-transformers"])
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--grouped-config", default=None, help="Optional JSON config path for grouped baseline profile.")
    p.add_argument("--domain-config", default=None, help="Optional JSON config path for domain-holdout baseline profile.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.artifact_dir

    grouped_base = (
        ECProtoNetV2Config.from_json(Path(args.grouped_config))
        if args.grouped_config
        else ECProtoNetV2Config(
            encoder=args.encoder,
            model_name=args.model_name,
            require_artifact_pass=False,
            require_active_contract=False,
        )
    )
    domain_base = (
        ECProtoNetV2Config.from_json(Path(args.domain_config))
        if args.domain_config
        else grouped_base
    )
    variants = {
        "default": {},
        "no_classifier_rescue": {"use_known_classifier": False},
        "classifier_rescue": {
            "use_known_classifier": True,
            "known_classifier_max_labels": 200,
            "known_label_allowlist": (),
            "min_support_per_aspect": 1,
        },
        "no_memory": {"use_memory": False},
        "review_queue_only": {"use_memory": True, "memory_source_mode": "review_queue", "memory_min_status": "detected"},
        "promoted_memory": {"use_memory": True, "memory_source_mode": "promoted", "memory_min_status": "promoted"},
        "illegal_oracle_memory": {"use_memory": True, "memory_source_mode": "oracle_all", "memory_min_status": "detected"},
        "no_open_world": {"emit_open_world_candidate": False},
        "no_equivalence_mapping": {"use_equivalence": False},
        "no_description_aliases": {
            "use_equivalence": False,
            "include_description_only_prototypes": False,
            "description_weight": 0.0,
            "train_evidence_weight": 1.0,
        },
        "no_prototype_head": {
            "w_proto": 0.0,
            "use_known_classifier": True,
            "known_classifier_max_labels": 200,
            "known_label_allowlist": (),
            "min_support_per_aspect": 1,
        },
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
        "confidence_only": {
            "open_world_unknown_residual_weight": 0.0,
            "open_world_unknown_energy_weight": 0.0,
            "open_world_unknown_confidence_weight": 1.0,
        },
        "residual_energy": {
            "open_world_unknown_residual_weight": 0.5,
            "open_world_unknown_energy_weight": 0.5,
            "open_world_unknown_confidence_weight": 0.0,
        },
        "evidence_low": {"w_evidence": 0.10},
        "evidence_high": {"w_evidence": 0.35},
        "no_aspect_graph": {"use_aspect_graph": False},
        "no_distilled_profiles": {"use_evidence_distilled_profiles": False},
        "no_graph_reranker_features": {"use_graph_reranker_features": False},
        "no_aspect_graph_no_profiles": {"use_aspect_graph": False, "use_evidence_distilled_profiles": False},
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
    report = _memory_effect_report(
        out / "no_memory" / "domain_holdout",
        out / "promoted_memory" / "domain_holdout",
    )
    write_json(out / "memory_effect_report.json", report)
    print(results)


if __name__ == "__main__":
    main()
