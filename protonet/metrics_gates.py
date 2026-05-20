from __future__ import annotations


def gate_candidate_ceiling(metrics: dict, min_topk_recall: float = 0.90) -> bool:
    return float(metrics.get("topk_recall", 0.0) or 0.0) >= float(min_topk_recall)


def gate_known_classifier_ready(metrics: dict, min_value: float = 0.80) -> bool:
    known = metrics.get("known_class_strict", {}) or {}
    return (
        float(known.get("precision", 0.0) or 0.0) >= min_value
        and float(known.get("recall", 0.0) or 0.0) >= min_value
        and float(known.get("f1", 0.0) or 0.0) >= min_value
    )


def gate_unknown_detector_ready(metrics: dict, min_auroc: float = 0.75, min_auprc: float = 0.60) -> bool:
    return (
        float(metrics.get("unknown_auroc", 0.0) or 0.0) >= min_auroc
        and float(metrics.get("unknown_auprc", 0.0) or 0.0) >= min_auprc
    )


def gate_router_not_main_bottleneck(metrics: dict, ratio_limit: float = 1.25) -> bool:
    buckets = metrics.get("error_buckets", {}) or {}
    rejected = float(buckets.get("candidate_found_but_router_rejected", 0) or 0)
    missing = float(buckets.get("candidate_missing_from_topk", 0) or 0)
    if missing <= 0:
        return rejected <= 0
    return rejected <= (missing * ratio_limit)
