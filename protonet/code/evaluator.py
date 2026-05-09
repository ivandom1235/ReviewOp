from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schema import ReviewExample, ECScore


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class ECEvaluator:
    def __init__(self):
        self.results: list[dict[str, Any]] = []

    def evaluate(
        self,
        examples: list[ReviewExample],
        predictions: list[list[ECScore]],
    ) -> dict[str, float]:
        tp_strict = fp_strict = fn_strict = 0
        tp_relaxed = rows_hit = total_valid_rows = 0

        # Selective / abstention counters
        accept_tp = accept_fp = 0
        abstain_tp = abstain_fp = abstain_fn = 0

        # Novelty counters
        novel_tp = novel_fp = novel_fn = 0

        # Decision distribution
        decision_dist: dict[str, int] = defaultdict(int)

        # Coverage (rows where model made ≥1 accepted prediction)
        covered_rows = 0

        for ex, preds in zip(examples, predictions):
            gold_aspects = {g.aspect for g in ex.gold_interpretations}
            gold_novel = {g.aspect for g in ex.gold_interpretations if g.novelty_status != "known"}

            accepted = [p for p in preds if p.decision in {"accept_known", "accept_multi_gold"}]
            abstained = [p for p in preds if p.decision == "abstain"]
            novel_routed = [p for p in preds if p.decision in {"open_world_candidate", "novel_candidate"}]

            for p in preds:
                decision_dist[p.decision] += 1

            if accepted:
                covered_rows += 1

            pred_aspects = {p.aspect for p in accepted}

            if not gold_aspects:
                fp_strict += len(pred_aspects)
                continue

            total_valid_rows += 1

            # Strict
            matched = gold_aspects & pred_aspects
            tp_strict += len(matched)
            fp_strict += len(pred_aspects - gold_aspects)
            fn_strict += len(gold_aspects - pred_aspects)

            # Relaxed (row-level)
            if matched:
                rows_hit += 1
                tp_relaxed += 1

            # Selective accepted accuracy
            for p in accepted:
                if p.aspect in gold_aspects:
                    accept_tp += 1
                else:
                    accept_fp += 1

            # Abstention: correct if model abstains and gold has no accepted aspect
            if not matched and ex.abstain_acceptable:
                if abstained:
                    abstain_tp += 1
                else:
                    abstain_fn += 1
            elif abstained and matched:
                abstain_fp += 1

            # Novelty
            novel_pred_aspects = {p.aspect for p in novel_routed}
            novel_matched = gold_novel & novel_pred_aspects
            novel_tp += len(novel_matched)
            novel_fp += len(novel_pred_aspects - gold_novel)
            novel_fn += len(gold_novel - novel_pred_aspects)

        precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
        recall_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
        f1_strict = compute_f1(precision_strict, recall_strict)

        relaxed_hit_rate = rows_hit / total_valid_rows if total_valid_rows > 0 else 0.0
        coverage = covered_rows / total_valid_rows if total_valid_rows > 0 else 0.0
        accepted_accuracy = accept_tp / (accept_tp + accept_fp) if (accept_tp + accept_fp) > 0 else 0.0

        abstain_prec = abstain_tp / (abstain_tp + abstain_fp) if (abstain_tp + abstain_fp) > 0 else 0.0
        abstain_rec = abstain_tp / (abstain_tp + abstain_fn) if (abstain_tp + abstain_fn) > 0 else 0.0
        abstain_f1 = compute_f1(abstain_prec, abstain_rec)

        novel_prec = novel_tp / (novel_tp + novel_fp) if (novel_tp + novel_fp) > 0 else 0.0
        novel_rec = novel_tp / (novel_tp + novel_fn) if (novel_tp + novel_fn) > 0 else 0.0
        novel_f1 = compute_f1(novel_prec, novel_rec)

        return {
            "strict_precision": precision_strict,
            "strict_recall": recall_strict,
            "strict_f1": f1_strict,
            "relaxed_multi_gold_f1": relaxed_hit_rate,
            "coverage": coverage,
            "accepted_accuracy": accepted_accuracy,
            "abstention_precision": abstain_prec,
            "abstention_recall": abstain_rec,
            "abstention_f1": abstain_f1,
            "novel_precision": novel_prec,
            "novel_recall": novel_rec,
            "novel_f1": novel_f1,
            "decision_distribution": dict(decision_dist),
            "total_rows_evaluated": float(len(examples)),
            "valid_gold_rows": float(total_valid_rows),
        }

    def evaluate_counterfactuals(
        self, pairs: list[dict], scorer: Any
    ) -> dict[str, float]:
        if not pairs:
            return {}

        consistent_count = 0
        aspect_swap_top1 = 0
        aspect_swap_top3 = 0
        sentiment_flip_success = 0
        total_valid = 0

        for pair in pairs:
            orig_text = pair.get("original_text")
            cf_text = pair.get("counterfactual_text")
            cf_gold = pair.get("expected_change", {}).get("aspect_changed_to")
            orig_gold = pair.get("expected_change", {}).get("aspect_changed_from")
            cf_type = pair.get("rewrite_type", "aspect_swap")

            if not orig_text or not cf_text:
                continue

            orig_ex = ReviewExample(
                row_id="orig", review_id="orig", text=orig_text, domain="unknown",
                split="test", source_type="explicit", novelty_status="known",
                abstain_acceptable=False, abstain_reason_gold=[], gold_interpretations=[],
            )
            cf_ex = ReviewExample(
                row_id="cf", review_id="cf", text=cf_text, domain="unknown",
                split="test", source_type="explicit", novelty_status="known",
                abstain_acceptable=False, abstain_reason_gold=[], gold_interpretations=[],
            )

            orig_preds = scorer.predict(orig_ex, top_k=3)
            cf_preds = scorer.predict(cf_ex, top_k=3)

            orig_top1 = orig_preds[0].aspect if orig_preds else None
            cf_top1 = cf_preds[0].aspect if cf_preds else None
            cf_top3 = {p.aspect for p in cf_preds}

            total_valid += 1

            if cf_type == "aspect_swap":
                if cf_top1 == cf_gold:
                    aspect_swap_top1 += 1
                    consistent_count += 1
                if cf_gold in cf_top3:
                    aspect_swap_top3 += 1
            elif cf_type == "sentiment_flip":
                if cf_top1 == orig_top1:
                    sentiment_flip_success += 1
                    consistent_count += 1

        return {
            "counterfactual_consistency": consistent_count / total_valid if total_valid > 0 else 0.0,
            "aspect_swap_top1_success_rate": aspect_swap_top1 / total_valid if total_valid > 0 else 0.0,
            "aspect_swap_top3_success_rate": aspect_swap_top3 / total_valid if total_valid > 0 else 0.0,
            "sentiment_flip_success_rate": sentiment_flip_success / total_valid if total_valid > 0 else 0.0,
        }
