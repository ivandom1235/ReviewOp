from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import ECProtoNetV2Config
from .label_normalizer import LabelNormalizer
from .schema import PredictionRecord


def _f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _relaxed_intersection(pred: set[str], gold: set[str], normalizer: LabelNormalizer) -> int:
    matched_gold: set[str] = set()
    count = 0
    for p in pred:
        for g in gold:
            if g not in matched_gold and normalizer.relaxed_match(p, g):
                matched_gold.add(g)
                count += 1
                break
    return count


def _record_pred_labels(record: PredictionRecord, config: ECProtoNetV2Config, mode: str = "final") -> list[str]:
    if mode == "final":
        return record.predicted_labels(config.accepted_decisions)
    if mode == "open_world_inclusive":
        return record.predicted_labels(config.open_world_accepted_decisions)
    if mode == "review_inclusive":
        return record.predicted_labels(config.review_accepted_decisions)
    raise ValueError(f"Unknown evaluation mode: {mode}")


def _coarse_parent(label: str) -> str:
    l = str(label or "").strip().lower()
    toks = [t for t in l.split("_") if t]
    if any(t in {"food", "dish", "meal", "dessert", "sushi", "pasta", "bagels", "chicken"} for t in toks):
        return "food_quality"
    if any(t in {"service", "staff", "waiter", "server"} for t in toks):
        return "service_quality"
    if any(t in {"speed", "wait", "delivery", "delay"} for t in toks):
        return "service_speed"
    if any(t in {"price", "value", "cost", "worth", "deal"} for t in toks):
        return "value"
    if any(t in {"ambience", "atmosphere", "decor", "vibe", "bar", "restaurant"} for t in toks):
        return "ambience"
    if any(t in {"battery", "power", "charge"} for t in toks):
        return "battery_life"
    if any(t in {"display", "screen", "resolution"} for t in toks):
        return "display"
    if any(t in {"keyboard", "key"} for t in toks):
        return "keyboard"
    if any(t in {"software", "app", "program", "windows"} for t in toks):
        return "software"
    if any(t in {"performance", "lag", "crash", "reliability"} for t in toks):
        return "performance"
    return l or "quality"


def multilabel_micro(
    records: list[PredictionRecord],
    normalizer: LabelNormalizer,
    relaxed: bool = False,
    *,
    config: ECProtoNetV2Config,
    mode: str = "final",
) -> dict[str, float]:
    tp = fp = fn = 0
    for r in records:
        pred = normalizer.normalize_set(_record_pred_labels(r, config, mode=mode))
        gold = normalizer.normalize_set(r.gold_labels)
        if relaxed:
            inter = _relaxed_intersection(pred, gold, normalizer)
        else:
            inter = len(pred & gold)
        tp += inter
        fp += max(0, len(pred) - inter)
        fn += max(0, len(gold) - inter)
    return _f1(tp, fp, fn)


def topk_micro(records: list[PredictionRecord], normalizer: LabelNormalizer, k: int, relaxed: bool = False) -> dict[str, float]:
    tp = fp = fn = 0
    for r in records:
        pred = normalizer.normalize_set([c.aspect for c in r.candidates[:k]])
        gold = normalizer.normalize_set(r.gold_labels)
        if relaxed:
            inter = _relaxed_intersection(pred, gold, normalizer)
        else:
            inter = len(pred & gold)
        tp += inter
        fp += max(0, len(pred) - inter)
        fn += max(0, len(gold) - inter)
    return _f1(tp, fp, fn)


def binary_f1(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return _f1(tp, fp, fn)


def accepted_accuracy(
    records: list[PredictionRecord],
    normalizer: LabelNormalizer,
    relaxed: bool = False,
    *,
    config: ECProtoNetV2Config,
    mode: str = "final",
) -> float:
    accepted = [r for r in records if _record_pred_labels(r, config, mode=mode)]
    if not accepted:
        return 0.0
    correct = 0
    for r in accepted:
        pred = normalizer.normalize_set(_record_pred_labels(r, config, mode=mode))
        gold = normalizer.normalize_set(r.gold_labels)
        ok = _relaxed_intersection(pred, gold, normalizer) > 0 if relaxed else bool(pred & gold)
        correct += int(ok)
    return correct / len(accepted)


def known_class_micro(
    records: list[PredictionRecord],
    normalizer: LabelNormalizer,
    relaxed: bool,
    *,
    config: ECProtoNetV2Config,
) -> dict[str, float]:
    tp = fp = fn = 0
    for r in records:
        pred = normalizer.normalize_set(
            label for label in _record_pred_labels(r, config, mode="final")
            if label != "__open_world__"
        )
        unseen = normalizer.normalize_set(r.gold_unseen_labels)
        gold_all = normalizer.normalize_set(r.gold_labels)
        gold = gold_all - unseen
        if relaxed:
            inter = _relaxed_intersection(pred, gold, normalizer)
        else:
            inter = len(pred & gold)
        tp += inter
        fp += max(0, len(pred) - inter)
        fn += max(0, len(gold) - inter)
    return _f1(tp, fp, fn)


def unseen_detection_f1(records: list[PredictionRecord]) -> dict[str, float]:
    y_true = [bool(r.gold_unseen_labels) for r in records]
    y_pred = [r.has_open_world for r in records]
    return binary_f1(y_true, y_pred)


def emerging_status_f1(records: list[PredictionRecord]) -> dict[str, float]:
    y_true = [bool(r.gold_emerging_labels or r.gold_novel_labels) for r in records]
    y_pred = [r.has_open_world for r in records]
    return binary_f1(y_true, y_pred)


def unknown_detection_auc(records: list[PredictionRecord]) -> dict[str, float | None]:
    y_true = [1 if r.gold_unseen_labels else 0 for r in records]
    y_score = [float(r.candidates[0].unknown_score) if r.candidates else 0.0 for r in records]
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


def risk_coverage(records: list[PredictionRecord], normalizer: LabelNormalizer, config: ECProtoNetV2Config) -> list[dict[str, float]]:
    pts: list[dict[str, float]] = []
    if not records:
        return pts
    ranked = sorted(
        records,
        key=lambda r: float(r.candidates[0].known_confidence) if r.candidates else 0.0,
        reverse=True,
    )
    for cutoff in (0.25, 0.50, 0.75, 1.00):
        n = max(1, int(round(len(ranked) * cutoff)))
        subset = ranked[:n]
        coverage = n / len(ranked)
        acc = accepted_accuracy(subset, normalizer, relaxed=False, config=config, mode="final")
        pts.append({"coverage": float(coverage), "risk": float(1.0 - acc)})
    return pts


def fixed_coverage_utility(
    records: list[PredictionRecord],
    normalizer: LabelNormalizer,
    config: ECProtoNetV2Config,
) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    ranked = sorted(
        records,
        key=lambda r: float(r.candidates[0].known_confidence) if r.candidates else 0.0,
        reverse=True,
    )
    out: dict[str, dict[str, float]] = {}
    for target in (0.50, 0.70, 0.90):
        n = max(1, int(round(len(ranked) * target)))
        subset = ranked[:n]
        achieved = n / len(ranked)
        acc = accepted_accuracy(subset, normalizer, relaxed=False, config=config, mode="final")
        out[f"{target:.2f}"] = {
            "coverage_target": float(target),
            "coverage_achieved": float(achieved),
            "accepted_accuracy_strict": float(acc),
            "risk": float(1.0 - acc),
            "utility": float(acc * achieved),
        }
    return out


def evaluate_predictions(records: list[PredictionRecord], normalizer: LabelNormalizer, config: ECProtoNetV2Config) -> dict:
    strict = multilabel_micro(records, normalizer, relaxed=False, config=config, mode="final")
    relaxed = multilabel_micro(records, normalizer, relaxed=True, config=config, mode="final")
    open_world_inclusive_strict = multilabel_micro(records, normalizer, relaxed=False, config=config, mode="open_world_inclusive")
    review_inclusive_strict = multilabel_micro(records, normalizer, relaxed=False, config=config, mode="review_inclusive")
    top1 = topk_micro(records, normalizer, k=1, relaxed=False)
    top3_relaxed = topk_micro(records, normalizer, k=min(3, config.top_k), relaxed=True)
    topk_relaxed = topk_micro(records, normalizer, k=config.top_k, relaxed=True)

    pred_any = [bool(_record_pred_labels(r, config, mode="final")) for r in records]
    coverage = sum(pred_any) / len(records) if records else 0.0

    abstain_gold = [r.abstain_acceptable for r in records]
    abstain_pred = [r.has_abstain for r in records]
    abstain = binary_f1(abstain_gold, abstain_pred)

    boundary_gold = [bool(r.gold_boundary_labels) for r in records]
    boundary_pred = [any(c.decision == "needs_review" for c in r.candidates[:2]) for r in records]
    boundary = binary_f1(boundary_gold, boundary_pred)

    known_strict = known_class_micro(records, normalizer, relaxed=False, config=config)
    known_relaxed = known_class_micro(records, normalizer, relaxed=True, config=config)
    unseen = unseen_detection_f1(records)
    unseen_auc = unknown_detection_auc(records)
    emerging = emerging_status_f1(records)

    decision_counts: dict[str, int] = {}
    for r in records:
        for c in r.candidates:
            if c.decision == "needs_review":
                continue
            decision_counts[c.decision] = decision_counts.get(c.decision, 0) + 1
    memory_hits = sum(
        1
        for r in records
        for c in r.candidates
        if float(getattr(c, "memory_support", 0.0) or 0.0) > 0.0
    )
    memory_hits_top1 = sum(
        1
        for r in records
        if r.candidates and float(getattr(r.candidates[0], "memory_support", 0.0) or 0.0) > 0.0
    )

    # Canonical/coarse-parent evaluation
    canon_tp = canon_fp = canon_fn = 0
    coarse_tp = coarse_fp = coarse_fn = 0
    domain_known_rows = 0
    domain_unseen_rows = 0
    known_only_tp = known_only_fp = known_only_fn = 0
    error_buckets = {
        "candidate_missing_from_topk": 0,
        "candidate_found_but_router_rejected": 0,
        "gold_label_not_in_train_inventory": 0,
        "gold_label_noise_or_fragment": 0,
        "canonical_mapping_missing": 0,
        "memory_should_have_matched": 0,
    }

    prototype_inventory = set()
    for r in records:
        for c in r.candidates:
            if c.aspect and c.aspect not in {"__open_world__", "unknown"}:
                prototype_inventory.add(normalizer.normalize(c.aspect))

    for r in records:
        pred_final = normalizer.normalize_set(_record_pred_labels(r, config, mode="final"))
        gold_final = normalizer.normalize_set(r.gold_labels)

        pred_canon = set(pred_final)
        gold_canon = set(gold_final)
        c_inter = pred_canon & gold_canon
        canon_tp += len(c_inter)
        canon_fp += max(0, len(pred_canon) - len(c_inter))
        canon_fn += max(0, len(gold_canon) - len(c_inter))

        pred_coarse = {_coarse_parent(x) for x in pred_canon}
        gold_coarse = {_coarse_parent(x) for x in gold_canon}
        p_inter = pred_coarse & gold_coarse
        coarse_tp += len(p_inter)
        coarse_fp += max(0, len(pred_coarse) - len(p_inter))
        coarse_fn += max(0, len(gold_coarse) - len(p_inter))

        unseen_labels = set(normalizer.normalize_set(r.gold_unseen_labels))
        known_gold = set(gold_canon) - unseen_labels
        if unseen_labels:
            domain_unseen_rows += 1
        else:
            domain_known_rows += 1
        known_pred = {x for x in pred_canon if x != "__open_world__"}
        k_inter = known_pred & known_gold
        known_only_tp += len(k_inter)
        known_only_fp += max(0, len(known_pred) - len(k_inter))
        known_only_fn += max(0, len(known_gold) - len(k_inter))

        if not known_gold:
            continue
        topk_labels = [normalizer.normalize(c.aspect) for c in r.candidates[: config.top_k] if c.aspect]
        accepted_labels = {normalizer.normalize(c.aspect) for c in r.candidates if c.decision == "accept_known" and c.aspect}
        for g in known_gold:
            if g in accepted_labels:
                continue
            if g in topk_labels:
                error_buckets["candidate_found_but_router_rejected"] += 1
            else:
                error_buckets["candidate_missing_from_topk"] += 1
            if g not in prototype_inventory:
                error_buckets["gold_label_not_in_train_inventory"] += 1
            toks = [t for t in g.split("_") if t]
            if len(toks) >= 3:
                error_buckets["gold_label_noise_or_fragment"] += 1
            if g == "unknown":
                error_buckets["canonical_mapping_missing"] += 1
            if r.candidates and float(getattr(r.candidates[0], "memory_support", 0.0) or 0.0) <= 0.0:
                error_buckets["memory_should_have_matched"] += 1

    return {
        "count": len(records),
        "strict": strict,
        "relaxed": relaxed,
        "top1_strict": top1,
        "top3_relaxed": top3_relaxed,
        "topk_relaxed": topk_relaxed,
        "topk_recall": topk_relaxed["recall"],
        "coverage": coverage,
        "accepted_accuracy_strict": accepted_accuracy(records, normalizer, relaxed=False, config=config, mode="final"),
        "accepted_accuracy_relaxed": accepted_accuracy(records, normalizer, relaxed=True, config=config, mode="final"),
        "final_strict_f1": strict["f1"],
        "final_relaxed_f1": relaxed["f1"],
        "known_class_strict": known_strict,
        "known_class_relaxed": known_relaxed,
        "unseen_detection": unseen,
        "unknown_auroc": unseen_auc["auroc"],
        "unknown_auprc": unseen_auc["auprc"],
        "risk_coverage": risk_coverage(records, normalizer, config),
        "fixed_coverage_utility": fixed_coverage_utility(records, normalizer, config),
        "emerging_status_open_world_alignment": emerging,
        "open_world_inclusive_strict": open_world_inclusive_strict,
        "review_inclusive_strict": review_inclusive_strict,
        "abstention": abstain,
        "true_novel": unseen,  # keep for backward compatibility
        "boundary": boundary,
        "decision_counts": decision_counts,
        "memory": {
            "source_mode": str(getattr(config, "memory_source_mode", "promoted")),
            "matches_used": int(memory_hits),
            "promoted_matches_used": int(memory_hits) if str(getattr(config, "memory_source_mode", "promoted")) == "promoted" else 0,
            "top1_matches_used": int(memory_hits_top1),
        },
        "canonical_strict": _f1(canon_tp, canon_fp, canon_fn),
        "coarse_parent_strict": _f1(coarse_tp, coarse_fp, coarse_fn),
        "domain_split_reporting": {
            "known_rows_count": int(domain_known_rows),
            "unseen_rows_count": int(domain_unseen_rows),
            "known_only_strict": _f1(known_only_tp, known_only_fp, known_only_fn),
        },
        "error_buckets": error_buckets,
    }


def objective(metrics: dict) -> float:
    coverage = float(metrics.get("coverage", 0.0))
    if coverage < 0.50:
        coverage_penalty = 0.50 - coverage
    elif coverage > 0.85:
        coverage_penalty = coverage - 0.85
    else:
        coverage_penalty = 0.0

    known_f1 = metrics["known_class_relaxed"]["f1"]
    unseen_f1 = metrics["unseen_detection"]["f1"]
    abstain_f1 = metrics["abstention"]["f1"]
    boundary_f1 = metrics["boundary"]["f1"]

    return (
        1.00 * known_f1
        + 0.50 * unseen_f1
        + 0.25 * abstain_f1
        + 0.25 * boundary_f1
        - 0.50 * coverage_penalty
    )
