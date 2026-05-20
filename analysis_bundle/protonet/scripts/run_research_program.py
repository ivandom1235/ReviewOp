from __future__ import annotations

import argparse
from pathlib import Path

from protonet.baselines import run_centroid_baseline, run_logreg_baseline
from protonet.dataset import load_dataset_bundle
from protonet.io_utils import read_jsonl, write_json
from protonet.label_normalizer import LabelNormalizer
from protonet.research_metrics import (
    bootstrap_ci,
    macro_f1_from_binary_matrix,
    micro_f1_from_sets,
    paired_bootstrap_delta,
)


def _proto_sets(predictions_path: Path, normalizer: LabelNormalizer) -> tuple[list[set[str]], list[set[str]]]:
    y_true: list[set[str]] = []
    y_pred: list[set[str]] = []
    for row in read_jsonl(predictions_path):
        y_true.append(normalizer.normalize_set(row.get("gold_labels") or []))
        y_pred.append(normalizer.normalize_set(row.get("predicted_labels") or []))
    return y_true, y_pred


def _auc(y_true: list[int], y_score: list[float]) -> dict[str, float | None]:
    if len(set(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        return {"auroc": None, "auprc": None}


def _summarize_baseline(name: str, y_true_sets, y_pred_sets, y_true_matrix, y_pred_matrix, y_true_unknown, y_score_unknown):
    return {
        "name": name,
        "micro_f1": micro_f1_from_sets(y_true_sets, y_pred_sets),
        "macro_f1": macro_f1_from_binary_matrix(y_true_matrix, y_pred_matrix),
        "unknown": _auc(y_true_unknown, y_score_unknown),
        "micro_f1_ci95": bootstrap_ci(micro_f1_from_sets, y_true_sets, y_pred_sets),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run broader research benchmark items (baseline + CI + paired bootstrap).")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--proto-grouped-preds", required=True)
    p.add_argument("--proto-domain-preds", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--encoder", default="hashing", choices=["hashing", "sentence-transformers"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--centroid-threshold", type=float, default=0.30)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset_bundle(args.artifact_dir)

    # Baseline grouped/domain-holdout
    base_grouped = run_logreg_baseline(
        bundle,
        bundle.test,
        train_rows=bundle.splits["train"],
        encoder_kind=args.encoder,
        threshold=args.threshold,
    )
    base_domain = run_logreg_baseline(
        bundle,
        bundle.domain_holdout["test"],
        train_rows=bundle.domain_holdout["train"],
        encoder_kind=args.encoder,
        threshold=args.threshold,
    )
    centroid_grouped = run_centroid_baseline(
        bundle,
        bundle.test,
        train_rows=bundle.splits["train"],
        encoder_kind=args.encoder,
        accept_threshold=args.centroid_threshold,
    )
    centroid_domain = run_centroid_baseline(
        bundle,
        bundle.domain_holdout["test"],
        train_rows=bundle.domain_holdout["train"],
        encoder_kind=args.encoder,
        accept_threshold=args.centroid_threshold,
    )

    # Proto predictions
    proto_g_true, proto_g_pred = _proto_sets(Path(args.proto_grouped_preds), bundle.normalizer)
    proto_d_true, proto_d_pred = _proto_sets(Path(args.proto_domain_preds), bundle.normalizer)

    # Grouped stats
    grouped = {
        "baselines": [
            _summarize_baseline(
                "logreg_ovr",
                base_grouped.y_true_sets,
                base_grouped.y_pred_sets,
                base_grouped.y_true_matrix,
                base_grouped.y_pred_matrix,
                base_grouped.y_true_unknown,
                base_grouped.y_score_unknown,
            ),
            _summarize_baseline(
                "centroid_top1",
                centroid_grouped.y_true_sets,
                centroid_grouped.y_pred_sets,
                centroid_grouped.y_true_matrix,
                centroid_grouped.y_pred_matrix,
                centroid_grouped.y_true_unknown,
                centroid_grouped.y_score_unknown,
            ),
        ],
        "proto_micro_f1": micro_f1_from_sets(proto_g_true, proto_g_pred),
        "proto_micro_f1_ci95": bootstrap_ci(micro_f1_from_sets, proto_g_true, proto_g_pred),
        "paired_delta_proto_minus_logreg": paired_bootstrap_delta(
            micro_f1_from_sets, base_grouped.y_true_sets, proto_g_pred, base_grouped.y_pred_sets
        ),
        "paired_delta_proto_minus_centroid": paired_bootstrap_delta(
            micro_f1_from_sets, centroid_grouped.y_true_sets, proto_g_pred, centroid_grouped.y_pred_sets
        ),
    }

    # Domain-holdout stats
    domain = {
        "baselines": [
            _summarize_baseline(
                "logreg_ovr",
                base_domain.y_true_sets,
                base_domain.y_pred_sets,
                base_domain.y_true_matrix,
                base_domain.y_pred_matrix,
                base_domain.y_true_unknown,
                base_domain.y_score_unknown,
            ),
            _summarize_baseline(
                "centroid_top1",
                centroid_domain.y_true_sets,
                centroid_domain.y_pred_sets,
                centroid_domain.y_true_matrix,
                centroid_domain.y_pred_matrix,
                centroid_domain.y_true_unknown,
                centroid_domain.y_score_unknown,
            ),
        ],
        "proto_micro_f1": micro_f1_from_sets(proto_d_true, proto_d_pred),
        "proto_micro_f1_ci95": bootstrap_ci(micro_f1_from_sets, proto_d_true, proto_d_pred),
        "paired_delta_proto_minus_logreg": paired_bootstrap_delta(
            micro_f1_from_sets, base_domain.y_true_sets, proto_d_pred, base_domain.y_pred_sets
        ),
        "paired_delta_proto_minus_centroid": paired_bootstrap_delta(
            micro_f1_from_sets, centroid_domain.y_true_sets, proto_d_pred, centroid_domain.y_pred_sets
        ),
    }

    report = {
        "artifact_dir": str(Path(args.artifact_dir).resolve()),
        "encoder": args.encoder,
        "threshold": args.threshold,
        "grouped_test": grouped,
        "domain_holdout_test": domain,
    }
    write_json(out / "research_program_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
