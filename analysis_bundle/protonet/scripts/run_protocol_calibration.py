from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

from protonet.config import ECProtoNetV2Config
from protonet.dataset import load_dataset_bundle
from protonet.evaluator import evaluate_predictions
from protonet.io_utils import write_json
from protonet.pipeline import build_context, predict_records, protocol_train_rows, run_compare


def _score(metrics: dict, protocol: str) -> float:
    strict = float(metrics.get("final_strict_f1", 0.0))
    coverage = float(metrics.get("coverage", 0.0))
    known = float(metrics.get("known_class_strict", {}).get("f1", 0.0))
    unknown_auprc = float(metrics.get("unknown_auprc") or 0.0)
    topk_recall = float(metrics.get("topk_recall") or 0.0)

    # Grouped protocol: optimize strict correctness first.
    if protocol == "grouped":
        return (1.30 * strict) + (0.45 * coverage) + (0.20 * known) + (0.20 * unknown_auprc)
    # Domain-holdout protocol: prioritize robustness/coverage under shift.
    return (0.95 * strict) + (0.85 * coverage) + (0.30 * topk_recall) + (0.10 * known)


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


def tune_protocol(bundle, protocol: str, base: ECProtoNetV2Config) -> dict:
    val = bundle.val if protocol == "grouped" else bundle.domain_holdout["val"]
    if not val:
        return {"best": None, "trials": 0, "top10": []}

    train_rows = protocol_train_rows(bundle, protocol)
    context = build_context(bundle, base, train_rows=train_rows)

    if protocol == "grouped":
        # Grouped profile: precision-oriented, controlled open-world emission.
        accept_grid = [0.32, 0.36, 0.40, 0.44, 0.48]
        abstain_grid = [0.10, 0.14, 0.18, 0.22]
        evidence_grid = [0.12, 0.18, 0.24, 0.30]
        unknown_grid = [0.45, 0.55, 0.65, 0.75]
        ow_qual_grid = [0.35, 0.45, 0.55]
        ow_ceil_grid = [0.25, 0.30, 0.35, 0.40]
    else:
        # Domain-holdout profile: recall/coverage-oriented under shift.
        accept_grid = [0.20, 0.24, 0.28, 0.32, 0.36]
        abstain_grid = [0.06, 0.10, 0.14, 0.18, 0.22]
        evidence_grid = [0.10, 0.16, 0.22, 0.28, 0.34]
        unknown_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        ow_qual_grid = [0.30, 0.40, 0.50, 0.60]
        ow_ceil_grid = [0.30, 0.35, 0.40, 0.45, 0.50]

    rows = []
    for a, ab, ev, ut, oq, oc in product(
        accept_grid, abstain_grid, evidence_grid, unknown_grid, ow_qual_grid, ow_ceil_grid
    ):
        if ab >= a:
            continue
        cfg = replace(
            base,
            accept_threshold=a,
            abstain_threshold=ab,
            evidence_abstain_threshold=ev,
            open_world_unknown_threshold=ut,
            open_world_evidence_quality_floor=oq,
            open_world_known_confidence_ceiling=oc,
            require_artifact_pass=False,
            require_active_contract=False,
        )
        # rebuild context with cfg priors retained from train
        c = replace(context, config=cfg)
        recs = predict_records(val, c)
        m = evaluate_predictions(recs, bundle.normalizer, cfg)
        rows.append({"objective": _score(m, protocol), "config": cfg.to_dict(), "metrics": m})

    rows.sort(key=lambda r: _sort_key(r, protocol), reverse=True)
    return {"best": rows[0] if rows else None, "trials": len(rows), "top10": rows[:10]}


def main() -> None:
    p = argparse.ArgumentParser(description="Tune grouped/domain-holdout thresholds independently.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(args.artifact_dir)
    base = ECProtoNetV2Config(require_artifact_pass=False, require_active_contract=False)

    grouped = tune_protocol(bundle, "grouped", base)
    domain = tune_protocol(bundle, "domain_holdout", base)

    write_json(out / "grouped_calibration.json", grouped)
    write_json(out / "domain_holdout_calibration.json", domain)

    summary = {"grouped": {}, "domain_holdout": {}}
    if grouped.get("best"):
        g_cfg_path = out / "grouped_best_config.json"
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
        d_cfg_path = out / "domain_holdout_best_config.json"
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

    write_json(out / "protocol_calibration_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
