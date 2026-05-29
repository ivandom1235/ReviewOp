from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import random

from protonet.config import ECProtoNetV2Config
from protonet.dataset import load_dataset_bundle
from protonet.evaluator import evaluate_predictions
from protonet.io_utils import write_json
from protonet.pipeline import build_context, predict_records, protocol_train_rows, run_compare
from protonet.schema import to_runtime_example
from protonet.scorer import compute_energy_stats

STABLE_ALLOWLIST = (
    "quality",
    "food_quality",
    "service_quality",
    "service_speed",
    "value",
    "price",
    "ambience",
    "cleanliness",
    "availability",
    "battery_life",
    "display",
    "keyboard",
    "trackpad",
    "storage",
    "software",
    "performance",
    "usability",
    "portability",
    "audio",
    "connectivity",
    "customer_support",
    "delivery",
    "design",
    "aesthetics",
    "comfort",
    "reliability",
    "power",
)


def _score(metrics: dict, protocol: str, mode: str = "known") -> float:
    strict = float(metrics.get("known_inventory", {}).get("strict", {}).get("f1", metrics.get("final_strict_f1", 0.0)))
    coverage = float(metrics.get("coverage", 0.0))
    known = float(metrics.get("known_inventory", {}).get("strict", {}).get("f1", metrics.get("known_class_strict", {}).get("f1", 0.0)))
    unknown_auprc = float(metrics.get("unknown_auprc") or 0.0)
    topk_recall = float(metrics.get("topk_recall") or 0.0)
    coverage_penalty = max(0.0, 0.50 - coverage)
    
    parent_label_f1 = float(metrics.get("parent_label_f1", 0.0))
    hierarchical_f1 = float(metrics.get("hierarchical_f1", 0.0))

    if mode == "open_world":
        unseen_f1 = float(metrics.get("unseen_detection", {}).get("f1", 0.0))
        auroc = float(metrics.get("unknown_auroc") or 0.0)
        auprc = float(metrics.get("unknown_auprc") or 0.0)
        return (0.40 * unseen_f1) + (0.30 * auroc) + (0.30 * auprc)
    if mode == "conservative":
        precision = float(metrics.get("known_inventory", {}).get("strict", {}).get("precision", metrics.get("precision", 0.0)))
        accepted_accuracy = float(metrics.get("accepted_accuracy_strict", 0.0))
        return (0.50 * precision) + (0.50 * accepted_accuracy)
    if mode == "known":
        return strict if protocol == "grouped" else known
    
    # joint/fallback
    if protocol == "grouped":
        return (1.10 * strict) + (0.25 * unknown_auprc) + (0.20 * topk_recall) - (0.50 * coverage_penalty) + (0.25 * parent_label_f1) + (0.25 * hierarchical_f1)
    return (1.00 * known) + (0.25 * unknown_auprc) + (0.20 * topk_recall) - (0.50 * coverage_penalty) + (0.25 * parent_label_f1) + (0.25 * hierarchical_f1)


def _sort_key(row: dict, protocol: str) -> tuple[float, float, float, float]:
    m = row["metrics"]
    objective = float(row["objective"])
    strict = float(m.get("final_strict_f1", 0.0))
    coverage = float(m.get("coverage", 0.0))
    topk_recall = float(m.get("topk_recall") or 0.0)
    unknown_auprc = float(m.get("unknown_auprc") or 0.0)
    if protocol == "grouped":
        return (objective, strict, unknown_auprc, coverage)
    return (objective, coverage, topk_recall, strict)


def tune_protocol(bundle, protocol: str, base: ECProtoNetV2Config, *, mode: str = "known", max_trials: int | None = None, seed: int = 13) -> dict:
    val = bundle.val if protocol == "grouped" else bundle.domain_holdout["val"]
    if not val:
        return {"best": None, "trials": 0, "top10": []}

    train_rows = protocol_train_rows(bundle, protocol)
    context = build_context(bundle, base, train_rows=train_rows)
    calibration_runtime = [to_runtime_example(ex) for ex in val]
    context.energy_stats = compute_energy_stats(calibration_runtime, context)

    if protocol == "grouped":
        accept_grid = [0.24, 0.28, 0.32, 0.36]
        abstain_grid = [0.06, 0.10, 0.14, 0.18]
        evidence_grid = [0.10, 0.16, 0.22, 0.28]
        proto_floor_grid = [0.10, 0.18, 0.25, 0.30]
        margin_grid = [0.00, 0.01, 0.03, 0.05]
        unknown_grid = [0.45, 0.55, 0.65, 0.75]
        ow_qual_grid = [0.35, 0.45, 0.55]
        ow_ceil_grid = [0.25, 0.30, 0.35, 0.40]
    else:
        accept_grid = [0.16, 0.20, 0.24, 0.28, 0.32]
        abstain_grid = [0.04, 0.08, 0.12, 0.16, 0.20]
        evidence_grid = [0.10, 0.16, 0.22, 0.28, 0.35]
        proto_floor_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
        margin_grid = [0.00, 0.01, 0.03, 0.05]
        unknown_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        ow_qual_grid = [0.30, 0.40, 0.50, 0.60]
        ow_ceil_grid = [0.30, 0.35, 0.40, 0.45, 0.50]

    ow_top1_proto_ceil_grid = [0.20, 0.30, 0.40]
    ow_margin_ceil_grid = [0.05, 0.15, 0.25]
    classifier_grid = [False, True]
    max_label_grid = [50, 100, 200]
    allowlist_grid = ["stable", "none"]
    min_support_grid = [1, 2, 3]

    combos = list(
        product(
            accept_grid,
            abstain_grid,
            evidence_grid,
            proto_floor_grid,
            margin_grid,
            unknown_grid,
            ow_qual_grid,
            ow_ceil_grid,
            ow_top1_proto_ceil_grid,
            ow_margin_ceil_grid,
            classifier_grid,
            max_label_grid,
            allowlist_grid,
            min_support_grid,
        )
    )

    # Keep calibration tractable by default even when no explicit max_trials is provided.
    trial_cap = max_trials if max_trials is not None and max_trials > 0 else 3000
    if len(combos) > trial_cap:
        rng = random.Random(seed)
        combos = rng.sample(combos, trial_cap)

    rows = []
    for a, ab, ev, pf, bm, ut, oq, oc, ow_top1_ceil, ow_margin_ceil, use_clf, max_labels, allowlist_mode, min_support in combos:
        if ab >= a:
            continue
        chosen_allowlist = () if allowlist_mode == "none" else STABLE_ALLOWLIST
        cfg = replace(
            base,
            use_known_classifier=use_clf,
            known_classifier_max_labels=max_labels,
            known_label_allowlist=chosen_allowlist,
            min_support_per_aspect=min_support,
            accept_threshold=a,
            implicit_accept_threshold=a,
            explicit_accept_threshold=max(a, min(0.36, a + 0.06)),
            abstain_threshold=ab,
            evidence_abstain_threshold=ev,
            known_label_evidence_floor=ev,
            implicit_known_label_evidence_floor=ev,
            explicit_known_label_evidence_floor=max(ev, min(0.35, ev + 0.08)),
            known_label_proto_floor=pf,
            boundary_margin_threshold=bm,
            open_world_unknown_threshold=ut,
            open_world_evidence_quality_floor=oq,
            open_world_known_confidence_ceiling=oc,
            open_world_top1_proto_ceiling=ow_top1_ceil,
            open_world_margin_ceiling=ow_margin_ceil,
            require_artifact_pass=False,
            require_active_contract=False,
        )
        # rebuild context with cfg priors retained from train
        c = replace(context, config=cfg)
        c.energy_stats = context.energy_stats
        recs = predict_records(val, c)
        m = evaluate_predictions(recs, bundle.normalizer, cfg)
        rows.append({"objective": _score(m, protocol, mode=mode), "config": cfg.to_dict(), "metrics": m})

    rows.sort(key=lambda r: _sort_key(r, protocol), reverse=True)
    return {"best": rows[0] if rows else None, "trials": len(rows), "top10": rows[:10]}


def main() -> None:
    p = argparse.ArgumentParser(description="Tune grouped/domain-holdout thresholds independently.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--encoder", default="hashing", choices=["hashing", "sentence-transformers"])
    p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max-trials", type=int, default=0, help="Optional cap to subsample grid-search trials per protocol.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--mode", choices=["known", "open_world", "conservative", "joint"], default="known")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(args.artifact_dir)
    base = ECProtoNetV2Config(
        encoder=args.encoder,
        model_name=args.model_name,
        require_artifact_pass=False,
        require_active_contract=False,
    )

    max_trials = args.max_trials if args.max_trials and args.max_trials > 0 else None
    grouped = tune_protocol(bundle, "grouped", base, mode=args.mode, max_trials=max_trials, seed=args.seed)
    domain = tune_protocol(bundle, "domain_holdout", base, mode=args.mode, max_trials=max_trials, seed=args.seed)

    write_json(out / f"grouped_calibration_{args.mode}.json", grouped)
    write_json(out / f"domain_holdout_calibration_{args.mode}.json", domain)

    summary = {"grouped": {}, "domain_holdout": {}}
    if grouped.get("best"):
        g_cfg_path = out / f"grouped_best_config_{args.mode}.json"
        write_json(g_cfg_path, grouped["best"]["config"])
        g = run_compare(
            artifact_dir=args.artifact_dir,
            output_dir=out / "grouped_test_with_grouped_tuned",
            config=ECProtoNetV2Config.from_json(g_cfg_path),
            protocol="grouped",
            split="test",
            allow_failed_artifact=True,
        )["metrics"]
        summary["grouped"] = {
            "test_final_strict_f1": g.get("final_strict_f1"),
            "test_coverage": g.get("coverage"),
            "test_unknown_auprc": g.get("unknown_auprc"),
        }
    if domain.get("best"):
        d_cfg_path = out / f"domain_holdout_best_config_{args.mode}.json"
        write_json(d_cfg_path, domain["best"]["config"])
        d = run_compare(
            artifact_dir=args.artifact_dir,
            output_dir=out / "domain_test_with_domain_tuned",
            config=ECProtoNetV2Config.from_json(d_cfg_path),
            protocol="domain_holdout",
            split="test",
            allow_failed_artifact=True,
        )["metrics"]
        summary["domain_holdout"] = {
            "test_final_strict_f1": d.get("final_strict_f1"),
            "test_coverage": d.get("coverage"),
            "test_unknown_auprc": d.get("unknown_auprc"),
        }

    summary["mode"] = args.mode
    write_json(out / f"protocol_calibration_summary_{args.mode}.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
