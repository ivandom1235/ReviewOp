from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def _gold_label(row: dict[str, Any]) -> tuple[str, str] | None:
    interpretations = list(row.get("implicit_grounded_interpretations") or [])
    if not interpretations:
        interpretations = list(row.get("gold_interpretations") or [])
    if not interpretations:
        return None
    item = interpretations[0]
    aspect = str(
        item.get("domain_canonical_aspect")
        or item.get("aspect_label")
        or item.get("aspect")
        or ""
    ).strip().lower()
    sentiment = str(item.get("sentiment") or "neutral").strip().lower()
    if not aspect:
        return None
    return aspect, sentiment


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _compute_metrics(rows: list[dict[str, Any]], predictions: dict[str, tuple[str, str]]) -> dict[str, Any]:
    labels = set()
    tp_total = fp_total = fn_total = 0
    per_label_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    domain_correct: Counter[str] = Counter()
    domain_total: Counter[str] = Counter()
    group_correct: Counter[str] = Counter()
    group_total: Counter[str] = Counter()

    for row in rows:
        row_id = str(row.get("instance_id") or row.get("record_id") or row.get("id") or "")
        gold = _gold_label(row)
        pred = predictions.get(row_id)
        domain = str(row.get("domain") or "unknown")
        group = str(row.get("group_id") or "unknown")

        if gold:
            labels.add(gold)
            domain_total[domain] += 1
            group_total[group] += 1
        if gold and pred == gold:
            domain_correct[domain] += 1
            group_correct[group] += 1

        if gold and pred:
            if pred == gold:
                per_label_stats[gold]["tp"] += 1
                tp_total += 1
            else:
                per_label_stats[pred]["fp"] += 1
                per_label_stats[gold]["fn"] += 1
                fp_total += 1
                fn_total += 1
        elif gold and not pred:
            per_label_stats[gold]["fn"] += 1
            fn_total += 1
        elif pred and not gold:
            per_label_stats[pred]["fp"] += 1
            fp_total += 1

    macro_scores = []
    for label in labels:
        stats = per_label_stats[label]
        macro_scores.append(_f1(stats["tp"], stats["fp"], stats["fn"]))

    micro_f1 = _f1(tp_total, fp_total, fn_total)
    macro_f1 = sum(macro_scores) / len(macro_scores) if macro_scores else 0.0
    domain_f1_proxy = {
        domain: (domain_correct[domain] / max(1, domain_total[domain]))
        for domain in domain_total
    }
    group_accuracy = {
        group: (group_correct[group] / max(1, group_total[group]))
        for group in group_total
    }
    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "worst_domain_f1": round(min(domain_f1_proxy.values()) if domain_f1_proxy else 0.0, 4),
        "worst_group_accuracy": round(min(group_accuracy.values()) if group_accuracy else 0.0, 4),
        "abstain_coverage": round(sum(1 for p in predictions.values() if p is None) / max(1, len(rows)), 4),
    }


def _build_erm_predictor(train_rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    domain_majority: dict[str, tuple[str, str]] = {}
    grouped: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in train_rows:
        label = _gold_label(row)
        if not label:
            continue
        domain = str(row.get("domain") or "unknown")
        grouped[domain][label] += 1
    for domain, counter in grouped.items():
        domain_majority[domain] = counter.most_common(1)[0][0]
    return domain_majority


def _build_groupdro_predictor(train_rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    # Proxy for group-DRO behavior: pick robust label per domain by inverse-group-frequency weighting.
    domain_scores: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    group_sizes = Counter(str(row.get("group_id") or "unknown") for row in train_rows)
    for row in train_rows:
        label = _gold_label(row)
        if not label:
            continue
        domain = str(row.get("domain") or "unknown")
        group = str(row.get("group_id") or "unknown")
        weight = 1.0 / max(1, group_sizes[group])
        domain_scores[domain][label] += weight

    predictors: dict[str, tuple[str, str]] = {}
    for domain, score_map in domain_scores.items():
        sorted_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        if sorted_items:
            predictors[domain] = sorted_items[0][0]
    return predictors


def evaluate_training_tracks(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train_rows = list(rows_by_split.get("train") or [])
    eval_rows = list(rows_by_split.get("val") or []) + list(rows_by_split.get("test") or [])

    erm_prior = _build_erm_predictor(train_rows)
    dro_prior = _build_groupdro_predictor(train_rows)

    erm_predictions = {
        str(row.get("instance_id") or row.get("record_id") or row.get("id") or ""): erm_prior.get(str(row.get("domain") or "unknown"))
        for row in eval_rows
    }
    dro_predictions = {
        str(row.get("instance_id") or row.get("record_id") or row.get("id") or ""): dro_prior.get(str(row.get("domain") or "unknown"))
        for row in eval_rows
    }

    return {
        "tracks": ["ERM", "GroupDRO"],
        "primary_protocols": ["grouped", "domain_holdout"],
        "secondary_protocols": ["random"],
        "erm": _compute_metrics(eval_rows, erm_predictions),
        "groupdro": _compute_metrics(eval_rows, dro_predictions),
    }


def promotion_gate(
    *,
    current_worst_domain_f1: float,
    previous_worst_domain_f1: float | None,
    max_regression: float = 0.02,
) -> dict[str, Any]:
    if previous_worst_domain_f1 is None:
        return {"worst_domain_regression": 0.0, "blocked": False, "reason": "no_previous_baseline"}
    regression = float(previous_worst_domain_f1) - float(current_worst_domain_f1)
    blocked = regression > max_regression
    return {
        "worst_domain_regression": round(regression, 4),
        "blocked": blocked,
        "reason": "worst_domain_f1_regressed" if blocked else "ok",
        "threshold": max_regression,
    }
