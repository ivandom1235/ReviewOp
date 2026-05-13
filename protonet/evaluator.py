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
    emerging = emerging_status_f1(records)

    decision_counts: dict[str, int] = {}
    for r in records:
        for c in r.candidates:
            decision_counts[c.decision] = decision_counts.get(c.decision, 0) + 1

    return {
        "count": len(records),
        "strict": strict,
        "relaxed": relaxed,
        "top1_strict": top1,
        "top3_relaxed": top3_relaxed,
        "topk_relaxed": topk_relaxed,
        "coverage": coverage,
        "accepted_accuracy_strict": accepted_accuracy(records, normalizer, relaxed=False, config=config, mode="final"),
        "accepted_accuracy_relaxed": accepted_accuracy(records, normalizer, relaxed=True, config=config, mode="final"),
        "final_strict_f1": strict["f1"],
        "final_relaxed_f1": relaxed["f1"],
        "known_class_strict": known_strict,
        "known_class_relaxed": known_relaxed,
        "unseen_detection": unseen,
        "emerging_status_open_world_alignment": emerging,
        "open_world_inclusive_strict": open_world_inclusive_strict,
        "review_inclusive_strict": review_inclusive_strict,
        "abstention": abstain,
        "true_novel": unseen,  # keep for backward compatibility
        "boundary": boundary,
        "decision_counts": decision_counts,
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
