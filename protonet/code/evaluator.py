from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import ReviewExample, ECScore


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class ECEvaluator:
    def __init__(self, equivalence_map_path: str | Path | None = None):
        self.results: list[dict[str, Any]] = []
        self.equivalence = self._load_equivalence_map(equivalence_map_path)

    @staticmethod
    def _normalize_aspect(value: Any) -> str:
        return str(value or "").strip().lower()

    def _load_equivalence_map(self, path: str | Path | None) -> dict[str, set[str]]:
        if path is None:
            path = Path(__file__).resolve().parent.parent / "configs" / "aspect_equivalence_map.json"
        p = Path(path)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
        eq: dict[str, set[str]] = {}
        if not isinstance(raw, dict):
            return eq
        for parent, aliases in raw.items():
            group = {self._normalize_aspect(parent)}
            if isinstance(aliases, list):
                group.update(self._normalize_aspect(v) for v in aliases)
            group = {g for g in group if g}
            for item in group:
                eq.setdefault(item, set()).update(group)
        return eq

    def _aspect_matches(self, pred: str, gold: str) -> bool:
        pred_n = self._normalize_aspect(pred)
        gold_n = self._normalize_aspect(gold)
        if pred_n == gold_n:
            return True
        pred_group = self.equivalence.get(pred_n, {pred_n})
        gold_group = self.equivalence.get(gold_n, {gold_n})
        return bool(pred_group & gold_group)

    def evaluate(
        self,
        examples: list[ReviewExample],
        predictions: list[list[ECScore]],
    ) -> dict[str, float]:
        tp_strict = fp_strict = fn_strict = 0
        tp_equiv = fp_equiv = fn_equiv = 0
        tp_relaxed = rows_hit = total_valid_rows = 0

        # Selective / abstention counters
        accept_tp = accept_fp = 0
        abstain_tp = abstain_fp = abstain_fn = 0

        # Novelty counters
        novel_tp = novel_fp = novel_fn = 0
        boundary_tp = boundary_fp = boundary_fn = 0

        # Decision distribution
        decision_dist: dict[str, int] = defaultdict(int)

        # Coverage (rows where model made ≥1 accepted prediction)
        covered_rows = 0
        covered_valid_rows = 0

        for ex, preds in zip(examples, predictions):
            gold_aspects = {g.aspect for g in ex.gold_interpretations}
            gold_true_novel = {g.aspect for g in ex.gold_interpretations if g.novelty_status == "novel"}
            gold_boundary = {g.aspect for g in ex.gold_interpretations if g.novelty_status == "boundary"}

            accepted = [p for p in preds if p.decision in {"accept_known", "accept_multi_gold", "unknown"}]
            abstained = [p for p in preds if p.decision == "abstain"]
            novel_routed = [p for p in preds if p.decision in {"open_world_candidate", "novel_candidate"}]

            for p in preds:
                decision_dist[p.decision] += 1

            if accepted:
                covered_rows += 1

            pred_aspects = {p.aspect for p in accepted}

            # Handle abstention/open-world rows before skipping no-gold rows
            # Abstention: correct if model abstains and gold has no accepted aspect (or row is intended for abstention)
            if not gold_aspects:
                if ex.abstain_acceptable:
                    if abstained:
                        abstain_tp += 1
                    else:
                        abstain_fn += 1
                elif abstained:
                    abstain_fp += 1
                
                fp_strict += len(pred_aspects)
                continue

            total_valid_rows += 1
            if accepted:
                covered_valid_rows += 1

            # Strict
            matched = gold_aspects & pred_aspects
            tp_strict += len(matched)
            fp_strict += len(pred_aspects - gold_aspects)
            fn_strict += len(gold_aspects - pred_aspects)

            # Equivalence-aware
            matched_pairs: set[tuple[str, str]] = set()
            used_preds: set[str] = set()
            for gold in gold_aspects:
                for pred in pred_aspects:
                    if pred in used_preds:
                        continue
                    if self._aspect_matches(pred, gold):
                        matched_pairs.add((pred, gold))
                        used_preds.add(pred)
                        break
            tp_equiv += len(matched_pairs)
            fp_equiv += len(pred_aspects) - len(used_preds)
            fn_equiv += len(gold_aspects) - len(matched_pairs)

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

            # Abstention for rows with gold aspects (should not abstain if there is a match)
            if abstained:
                abstain_fp += 1
            elif ex.abstain_acceptable and not matched:
                abstain_fn += 1

            # Novelty
            novel_pred_aspects = {p.aspect for p in novel_routed}
            novel_matched = gold_true_novel & novel_pred_aspects
            novel_tp += len(novel_matched)
            novel_fp += len(novel_pred_aspects - gold_true_novel)
            novel_fn += len(gold_true_novel - novel_pred_aspects)

            boundary_routed = [p for p in preds if p.decision in {"needs_review", "open_world_candidate", "abstain"}]
            boundary_pred_aspects = {p.aspect for p in boundary_routed}
            boundary_tp += len(gold_boundary & boundary_pred_aspects)
            boundary_fp += len(boundary_pred_aspects - gold_boundary)
            boundary_fn += len(gold_boundary - boundary_pred_aspects)

        precision_strict = tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
        recall_strict = tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
        f1_strict = compute_f1(precision_strict, recall_strict)
        precision_equiv = tp_equiv / (tp_equiv + fp_equiv) if (tp_equiv + fp_equiv) > 0 else 0.0
        recall_equiv = tp_equiv / (tp_equiv + fn_equiv) if (tp_equiv + fn_equiv) > 0 else 0.0
        f1_equiv = compute_f1(precision_equiv, recall_equiv)

        relaxed_hit_rate = rows_hit / total_valid_rows if total_valid_rows > 0 else 0.0
        
        # Fixed coverage calculation
        total_rows = len(examples)
        coverage = covered_rows / max(1, total_rows)
        coverage_on_valid_gold_rows = covered_valid_rows / max(1, total_valid_rows)
        
        accepted_accuracy = accept_tp / (accept_tp + accept_fp) if (accept_tp + accept_fp) > 0 else 0.0

        abstain_prec = abstain_tp / (abstain_tp + abstain_fp) if (abstain_tp + abstain_fp) > 0 else 0.0
        abstain_rec = abstain_tp / (abstain_tp + abstain_fn) if (abstain_tp + abstain_fn) > 0 else 0.0
        abstain_f1 = compute_f1(abstain_prec, abstain_rec)

        novel_prec = novel_tp / (novel_tp + novel_fp) if (novel_tp + novel_fp) > 0 else 0.0
        novel_rec = novel_tp / (novel_tp + novel_fn) if (novel_tp + novel_fn) > 0 else 0.0
        novel_f1 = compute_f1(novel_prec, novel_rec)
        boundary_prec = boundary_tp / (boundary_tp + boundary_fp) if (boundary_tp + boundary_fp) > 0 else 0.0
        boundary_rec = boundary_tp / (boundary_tp + boundary_fn) if (boundary_tp + boundary_fn) > 0 else 0.0
        boundary_f1 = compute_f1(boundary_prec, boundary_rec)

        return {
            "strict_precision": precision_strict,
            "strict_recall": recall_strict,
            "strict_f1": f1_strict,
            "strict_exact_f1": f1_strict,
            "equivalence_relaxed_precision": precision_equiv,
            "equivalence_relaxed_recall": recall_equiv,
            "equivalence_relaxed_f1": f1_equiv,
            "relaxed_multi_gold_f1": relaxed_hit_rate,
            "coverage": coverage,
            "coverage_on_valid_gold_rows": coverage_on_valid_gold_rows,
            "accepted_accuracy": accepted_accuracy,
            "abstention_precision": abstain_prec,
            "abstention_recall": abstain_rec,
            "abstention_f1": abstain_f1,
            "novel_precision": novel_prec,
            "novel_recall": novel_rec,
            "novel_f1": novel_f1,
            "true_novel_f1": novel_f1,
            "boundary_handling_f1": boundary_f1,
            "decision_distribution": dict(decision_dist),
            "total_rows_evaluated": float(total_rows),
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
        
        aspect_swap_total = 0
        sentiment_flip_total = 0
        total_valid = 0

        def norm(s: Any) -> str:
            return str(s or "").strip().lower()

        for pair in pairs:
            orig_text = pair.get("original_text")
            cf_text = pair.get("counterfactual_text")
            
            # Robust schema parsing
            cf_gold = pair.get("target_aspect") or pair.get("expected_change", {}).get("aspect_changed_to")
            orig_gold = pair.get("source_aspect") or pair.get("expected_change", {}).get("aspect_changed_from")
            
            # rewrite_type or counterfactual_type
            cf_type = pair.get("counterfactual_type") or pair.get("rewrite_type", "aspect_swap")

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
                aspect_swap_total += 1
                if norm(cf_top1) == norm(cf_gold):
                    aspect_swap_top1 += 1
                    consistent_count += 1
                if any(norm(a) == norm(cf_gold) for a in cf_top3):
                    aspect_swap_top3 += 1
            elif cf_type == "sentiment_flip":
                sentiment_flip_total += 1
                # Sentiment flip: aspect should stay the same
                if norm(cf_top1) == norm(orig_top1) and cf_top1 is not None:
                    sentiment_flip_success += 1
                    consistent_count += 1

        return {
            "counterfactual_consistency": consistent_count / total_valid if total_valid > 0 else 0.0,
            "aspect_swap_top1_success_rate": aspect_swap_top1 / max(1, aspect_swap_total) if aspect_swap_total > 0 else 0.0,
            "aspect_swap_top3_success_rate": aspect_swap_top3 / max(1, aspect_swap_total) if aspect_swap_total > 0 else 0.0,
            "sentiment_flip_success_rate": sentiment_flip_success / max(1, sentiment_flip_total) if sentiment_flip_total > 0 else 0.0,
            "aspect_swap_total": float(aspect_swap_total),
            "sentiment_flip_total": float(sentiment_flip_total),
        }

