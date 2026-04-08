from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, List
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

try:
    from .config import ProtonetConfig
    from .progress import task_bar
except ImportError:
    _config_path = Path(__file__).resolve().with_name("config.py")
    _config_spec = importlib.util.spec_from_file_location("protonet_local_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:  # pragma: no cover
        raise
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    ProtonetConfig = _config_module.ProtonetConfig
    from progress import task_bar


def _aspect_from_joint(label: str, separator: str) -> str:
    return label.split(separator, 1)[0]


def _expected_calibration_error(confidences: List[float], correct: List[int], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (conf >= left) & (conf < right if i < bins - 1 else conf <= right)
        if not mask.any():
            continue
        bucket_conf = conf[mask].mean()
        bucket_acc = corr[mask].mean()
        ece += abs(bucket_acc - bucket_conf) * (mask.sum() / len(conf))
    return float(ece)


def _stable_cluster_id(*, domain: str, hint: str) -> str:
    basis = f"v2-novel-cluster|{str(domain).strip().lower()}|{str(hint).strip().lower()}"
    return f"novel_{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]}"


def _decode_post_aspect_prediction(
    probabilities: np.ndarray,
    ordered_labels: List[str],
    *,
    separator: str,
    multi_label_margin: float,
) -> dict[str, Any]:
    aspect_best: dict[str, tuple[float, int]] = {}
    for idx, label in enumerate(ordered_labels):
        aspect = _aspect_from_joint(label, separator)
        score = float(probabilities[idx])
        current = aspect_best.get(aspect)
        if current is None or score > current[0]:
            aspect_best[aspect] = (score, idx)
    if not aspect_best:
        return {
            "pred_label": "",
            "pred_labels": [],
            "confidence": 0.0,
            "selected_aspects": [],
            "aspect": "",
            "sentiment": "neutral",
        }

    ranked = sorted(aspect_best.items(), key=lambda item: item[1][0], reverse=True)
    top_score = ranked[0][1][0]
    selected = [item for item in ranked if item[1][0] >= top_score - float(multi_label_margin)]
    selected.sort(key=lambda item: item[1][0], reverse=True)
    pred_labels = [ordered_labels[idx] for _, (_, idx) in selected]
    best_aspect, (best_score, best_index) = ranked[0]
    best_label = ordered_labels[best_index]
    aspect = _aspect_from_joint(best_label, separator)
    sentiment = best_label.split(separator, 1)[1] if separator in best_label else "neutral"
    return {
        "pred_label": best_label,
        "pred_labels": pred_labels,
        "confidence": float(best_score),
        "selected_aspects": [aspect_name for aspect_name, _ in selected],
        "aspect": aspect or best_aspect,
        "sentiment": sentiment,
    }


def _project_prediction_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "joint":
        return list(rows)
    if mode != "post_aspect":
        return list(rows)
    projected: List[Dict[str, Any]] = []
    for row in rows:
        pred_label = str(row.get("post_aspect_pred_label") or row.get("pred_label") or "")
        pred_labels = list(row.get("post_aspect_pred_labels") or ([pred_label] if pred_label else []))
        projected.append(
            {
                **row,
                "pred_label": pred_label,
                "pred_labels": pred_labels,
                "confidence": float(row.get("post_aspect_confidence", row.get("confidence", 0.0))),
                "correct": bool(row.get("post_aspect_correct", row.get("correct", False))),
                "flex_correct": bool(row.get("post_aspect_flex_correct", row.get("flex_correct", False))),
                "multi_label_overlap": float(row.get("post_aspect_multi_label_overlap", row.get("multi_label_overlap", 0.0))),
                "abstained": bool(row.get("post_aspect_abstained", row.get("abstained", False))),
                "low_confidence": bool(row.get("post_aspect_low_confidence", row.get("low_confidence", False))),
            }
        )
    return projected


def _compact_mode_metrics(
    rows: List[Dict[str, Any]],
    cfg: ProtonetConfig,
    split_name: str,
    *,
    mode: str,
    elapsed: float,
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not rows:
        return {
            "split": split_name,
            "num_episodes": len(episodes),
            "num_queries": 0,
            "accuracy": 0.0,
            "aspect_only_accuracy": 0.0,
            "macro_f1": 0.0,
            "flexible_match_score": 0.0,
            "multi_label_overlap_score": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "abstention_f1": 0.0,
            "coverage": 0.0,
            "risk": 0.0,
            "low_confidence_rate": 0.0,
            "selected_mode": mode,
        }
    y_true = [str(row.get("true_label") or "") for row in rows]
    y_pred = [str(row.get("pred_label") or "") for row in rows]
    y_true_aspect = [_aspect_from_joint(label, cfg.joint_label_separator) for label in y_true]
    y_pred_aspect = [_aspect_from_joint(label, cfg.joint_label_separator) for label in y_pred]
    correctness = [int(row.get("correct", False)) for row in rows]
    low_confidence_count = sum(1 for row in rows if bool(row.get("abstained", False)))
    abstain_true_positive = sum(1 for row in rows if bool(row.get("abstained", False)) and not bool(row.get("correct", False)))
    abstain_false_positive = sum(1 for row in rows if bool(row.get("abstained", False)) and bool(row.get("correct", False)))
    abstain_false_negative = sum(1 for row in rows if not bool(row.get("abstained", False)) and not bool(row.get("correct", False)))
    confidence_error = _expected_calibration_error([float(row.get("confidence", 0.0)) for row in rows], correctness)
    coverage = 1.0 - (low_confidence_count / max(1, len(rows)))
    risk = 1.0 - (sum(correctness) / max(1, len(rows)))
    return {
        "split": split_name,
        "num_episodes": len(episodes),
        "num_queries": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "aspect_only_accuracy": float(accuracy_score(y_true_aspect, y_pred_aspect)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "flexible_match_score": float(np.mean([1.0 if row.get("flex_correct") else 0.0 for row in rows])) if rows else 0.0,
        "multi_label_overlap_score": float(np.mean([float(row.get("multi_label_overlap", 0.0)) for row in rows])) if rows else 0.0,
        "abstention_precision": abstain_true_positive / max(1, abstain_true_positive + abstain_false_positive),
        "abstention_recall": abstain_true_positive / max(1, abstain_true_positive + abstain_false_negative),
        "abstention_f1": (
            2
            * (abstain_true_positive / max(1, abstain_true_positive + abstain_false_positive))
            * (abstain_true_positive / max(1, abstain_true_positive + abstain_false_negative))
            / max(
                1e-12,
                (abstain_true_positive / max(1, abstain_true_positive + abstain_false_positive))
                + (abstain_true_positive / max(1, abstain_true_positive + abstain_false_negative)),
            )
        ),
        "coverage": float(coverage),
        "risk": float(risk),
        "low_confidence_rate": low_confidence_count / max(1, len(rows)),
        "calibration_ece": float(confidence_error),
        "selected_mode": mode,
    }


def evaluate_episodes(model, episodes: List[Dict[str, Any]], cfg: ProtonetConfig, split_name: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    eval_temperature = float(model.temperature.detach().cpu().item())
    y_true: List[str] = []
    y_pred: List[str] = []
    y_true_aspect: List[str] = []
    y_pred_aspect: List[str] = []
    confidences: List[float] = []
    correctness: List[int] = []
    low_confidence_count = 0
    abstain_true_positive = 0
    abstain_false_positive = 0
    abstain_false_negative = 0
    running_correct = 0
    per_aspect: Dict[str, List[int]] = defaultdict(list)
    predictions: List[Dict[str, Any]] = []
    novelty_truth: List[int] = []
    novelty_scores: List[float] = []
    novelty_pred: List[int] = []

    start_time = time.perf_counter()
    with torch.no_grad():
        with task_bar(total=len(episodes), desc=f"eval:{split_name}", enabled=cfg.progress_enabled) as bar:
            for episode in episodes:
                out = model.episode_forward(episode)
                probs = out.probabilities.numpy()
                target_indices = out.targets.detach().cpu().tolist()
                query_rows = list(episode.get("query_set", []))
                for row_index, target_idx in enumerate(target_indices):
                    true_label = out.ordered_labels[target_idx]
                    pred_label = out.predictions[row_index]
                    post_aspect = _decode_post_aspect_prediction(
                        probs[row_index],
                        out.ordered_labels,
                        separator=cfg.joint_label_separator,
                        multi_label_margin=cfg.multi_label_margin,
                    )
                    post_pred_label = str(post_aspect.get("pred_label") or pred_label)
                    post_pred_labels = list(post_aspect.get("pred_labels") or ([post_pred_label] if post_pred_label else []))
                    confidence = float(probs[row_index].max())
                    top_indices = np.argsort(probs[row_index])[::-1][: cfg.top_k_debug].tolist()
                    top_predictions = [
                        {
                            "label": out.ordered_labels[idx],
                            "probability": float(probs[row_index][idx]),
                        }
                        for idx in top_indices
                    ]
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    y_true_aspect.append(_aspect_from_joint(true_label, cfg.joint_label_separator))
                    y_pred_aspect.append(_aspect_from_joint(pred_label, cfg.joint_label_separator))
                    confidences.append(confidence)
                    is_correct = int(true_label == pred_label)
                    correctness.append(is_correct)
                    running_correct += is_correct
                    did_abstain = confidence < cfg.low_confidence_threshold
                    if did_abstain:
                        low_confidence_count += 1
                        if not is_correct:
                            abstain_true_positive += 1
                        else:
                            abstain_false_positive += 1
                    elif not is_correct:
                        abstain_false_negative += 1
                    per_aspect[_aspect_from_joint(true_label, cfg.joint_label_separator)].append(is_correct)
                    query_row = query_rows[row_index] if row_index < len(query_rows) else {}
                    interpretations = list(query_row.get("gold_interpretations") or []) if isinstance(query_row, dict) else []
                    if interpretations:
                        true_set = {f"{it.get('aspect_label')}{cfg.joint_label_separator}{it.get('sentiment')}" for it in interpretations if isinstance(it, dict)}
                    else:
                        gold_joint_labels = list(query_row.get("gold_joint_labels") or []) if isinstance(query_row, dict) else []
                        true_set = set(gold_joint_labels) if gold_joint_labels else {true_label}
                    
                    pred_set = {pred_label}
                    jaccard = len(true_set & pred_set) / max(1, len(true_set | pred_set))
                    row_logits = out.logits[row_index].detach().cpu()
                    max_logit = float(row_logits.max().item())
                    min_distance_sq = max(0.0, -max_logit * eval_temperature)
                    distance_score = max(0.0, min(1.0, min_distance_sq / (min_distance_sq + 1.0)))
                    ranked_probs = np.sort(probs[row_index])[::-1]
                    p_top1 = float(ranked_probs[0]) if len(ranked_probs) > 0 else 0.0
                    p_top2 = float(ranked_probs[1]) if len(ranked_probs) > 1 else 0.0
                    ambiguity_score = max(0.0, min(1.0, 1.0 - (p_top1 - p_top2)))
                    energy_raw = float((-eval_temperature * torch.logsumexp(row_logits, dim=0)).item())
                    energy_score = max(0.0, min(1.0, (energy_raw + 5.0) / 10.0))
                    # Change 21: Improved novelty scoring (re-weighted to sum to 1.0)
                    novelty_score = max(
                        0.0,
                        min(
                            1.0,
                            0.50 * distance_score + 0.30 * ambiguity_score + 0.20 * energy_score,
                        ),
                    )
                    split_protocol = query_row.get("split_protocol") if isinstance(query_row, dict) else {}
                    novel_truth_label = 1 if bool(query_row.get("novel_acceptable", False)) else 0
                    novelty_truth.append(novel_truth_label)
                    novelty_scores.append(float(novelty_score))
                    decision_band = "known"
                    if novelty_score >= float(cfg.novelty_novel_threshold):
                        decision_band = "novel"
                    elif novelty_score > float(cfg.novelty_known_threshold):
                        decision_band = "boundary"
                    routing = "novel" if decision_band == "novel" else "known"
                    novelty_pred.append(1 if routing == "novel" else 0)
                    evidence_hint = ""
                    domain_hint = "unknown"
                    if isinstance(query_row, dict):
                        evidence_hint = str(
                            query_row.get("novel_evidence_text")
                            or query_row.get("evidence_text")
                            or query_row.get("review_text")
                            or ""
                        )
                        domain_hint = str(query_row.get("domain") or "unknown")
                    predicted_cluster_id = (
                        _stable_cluster_id(domain=domain_hint, hint=evidence_hint)
                        if routing == "novel"
                        else None
                    )
                    predictions.append(
                        {
                            "episode_id": episode.get("episode_id"),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "post_aspect_pred_label": post_pred_label,
                            "post_aspect_pred_labels": post_pred_labels,
                            "post_aspect_confidence": float(post_aspect.get("confidence", confidence)),
                            "confidence": confidence,
                            "low_confidence": confidence < cfg.low_confidence_threshold,
                            "post_aspect_low_confidence": float(post_aspect.get("confidence", confidence)) < cfg.low_confidence_threshold,
                            "top_k": top_predictions,
                            "correct": bool(is_correct),
                            "post_aspect_correct": bool(true_label == post_pred_label),
                            "split": split_name,
                            "flex_correct": bool(pred_label in true_set),
                            "post_aspect_flex_correct": bool(set(post_pred_labels) & true_set),
                            "multi_label_overlap": float(jaccard),
                            "post_aspect_multi_label_overlap": float(len(true_set & set(post_pred_labels)) / max(1, len(true_set | set(post_pred_labels)))),
                            "abstained": did_abstain,
                            "post_aspect_abstained": float(post_aspect.get("confidence", confidence)) < cfg.low_confidence_threshold,
                            "novelty_score": float(novelty_score),
                            "novelty_components": {
                                "distance_score": float(distance_score),
                                "ambiguity_score": float(ambiguity_score),
                                "energy_score": float(energy_score),
                            },
                            "routing": routing,
                            "decision_band": decision_band,
                            "split_protocol": split_protocol if isinstance(split_protocol, dict) else {},
                            "benchmark_ambiguity_score": float(query_row.get("benchmark_ambiguity_score", 0.0)) if isinstance(query_row, dict) else 0.0,
                            "abstain_acceptable": bool(query_row.get("abstain_acceptable", False)) if isinstance(query_row, dict) else False,
                            "ambiguity_type": query_row.get("ambiguity_type") if isinstance(query_row, dict) else None,
                            "novel_acceptable": bool(query_row.get("novel_acceptable", False)) if isinstance(query_row, dict) else False,
                            "gold_novel_cluster_id": str(query_row.get("novel_cluster_id") or "").strip() if isinstance(query_row, dict) else "",
                            "pred_novel_cluster_id": predicted_cluster_id,
                            "post_aspect_selected_aspects": list(post_aspect.get("selected_aspects") or []),
                            "post_aspect_sentiment": str(post_aspect.get("sentiment") or "neutral"),
                        }
                    )
                bar.update(1)
                if y_true:
                    bar.set_postfix(
                        acc=f"{running_correct / len(y_true):.3f}",
                        queries=len(y_true),
                        low_conf=f"{low_confidence_count / max(1, len(y_true)):.3f}",
                    )

    elapsed = time.perf_counter() - start_time
    abstain_precision = abstain_true_positive / max(1, abstain_true_positive + abstain_false_positive)
    abstain_recall = abstain_true_positive / max(1, abstain_true_positive + abstain_false_negative)
    abstain_f1 = (2 * abstain_precision * abstain_recall / (abstain_precision + abstain_recall)) if (abstain_precision + abstain_recall) else 0.0
    coverage = 1.0 - (low_confidence_count / max(1, len(y_true)))
    risk = 1.0 - (running_correct / max(1, len(y_true)))
    avg_overlap = float(np.mean([float(row.get("multi_label_overlap", 0.0)) for row in predictions])) if predictions else 0.0
    flex_correct_rate = float(np.mean([1.0 if row.get("flex_correct") else 0.0 for row in predictions])) if predictions else 0.0
    has_novel_positives = any(int(value) == 1 for value in novelty_truth)
    known_novel_quality = float(np.mean([1.0 if int(pred) == int(truth) else 0.0 for pred, truth in zip(novelty_pred, novelty_truth)])) if novelty_truth else 0.0
    known_vs_novel_auroc = 0.0
    if has_novel_positives and len(set(novelty_truth)) > 1:
        try:
            known_vs_novel_auroc = float(roc_auc_score(novelty_truth, novelty_scores))
        except ValueError:
            known_vs_novel_auroc = 0.0
    tp = sum(1 for pred, truth in zip(novelty_pred, novelty_truth) if pred == 1 and truth == 1)
    fp = sum(1 for pred, truth in zip(novelty_pred, novelty_truth) if pred == 1 and truth == 0)
    fn = sum(1 for pred, truth in zip(novelty_pred, novelty_truth) if pred == 0 and truth == 1)
    tn = sum(1 for pred, truth in zip(novelty_pred, novelty_truth) if pred == 0 and truth == 0)
    precision_novel = tp / max(1, tp + fp)
    recall_novel = tp / max(1, tp + fn)
    f1_novel = (2 * precision_novel * recall_novel / (precision_novel + recall_novel)) if (precision_novel + recall_novel) else 0.0
    precision_known = tn / max(1, tn + fn)
    recall_known = tn / max(1, tn + fp)
    f1_known = (2 * precision_known * recall_known / (precision_known + recall_known)) if (precision_known + recall_known) else 0.0
    known_novel_f1_macro = (f1_known + f1_novel) / 2.0 if has_novel_positives else 0.0
    boundary_rows = [row for row in predictions if str(row.get("decision_band") or "") == "boundary"]
    boundary_abstain_quality = float(
        np.mean(
            [
                1.0
                if (bool(row.get("abstained", False)) or bool(row.get("abstain_acceptable", False)))
                else 0.0
                for row in boundary_rows
            ]
        )
    ) if boundary_rows else 0.0
    cluster_pairs = [
        row
        for row in predictions
        if bool(row.get("novel_acceptable", False))
        and str(row.get("routing") or "") == "novel"
        and str(row.get("gold_novel_cluster_id") or "").strip()
    ]
    cluster_consistency = float(
        np.mean(
            [
                1.0 if str(row.get("pred_novel_cluster_id") or "") == str(row.get("gold_novel_cluster_id") or "") else 0.0
                for row in cluster_pairs
            ]
        )
    ) if cluster_pairs else 0.0
    open_set_curve = []
    if predictions:
        quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]
        novelty_array = np.asarray(novelty_scores if novelty_scores else [0.0], dtype=float)
        for q in quantiles:
            threshold = float(np.quantile(novelty_array, q))
            covered = [row for row in predictions if float(row.get("novelty_score", 0.0)) <= threshold]
            coverage_q = len(covered) / max(1, len(predictions))
            risk_q = 1.0 - (sum(1 for row in covered if bool(row.get("correct"))) / max(1, len(covered)))
            open_set_curve.append({"quantile": q, "threshold": threshold, "coverage": float(coverage_q), "risk": float(risk_q)})
    high_ambiguity_values = [1.0 if row.get("correct") else 0.0 for row in predictions if float(row.get("benchmark_ambiguity_score", 0.0)) >= 0.5]
    high_ambiguity_accuracy = float(np.mean(high_ambiguity_values)) if high_ambiguity_values else 0.0
    protocol_groups: Dict[str, List[int]] = {"random": [], "grouped": [], "domain_holdout": []}
    for idx, row in enumerate(predictions):
        protocol_payload = row.get("split_protocol") or {}
        grouped_value = protocol_payload.get("grouped", protocol_payload.get("source_holdout"))
        normalized_protocol_payload = {
            "random": protocol_payload.get("random"),
            "grouped": grouped_value,
            "domain_holdout": protocol_payload.get("domain_holdout"),
        }
        for protocol in ("random", "grouped", "domain_holdout"):
            if str(normalized_protocol_payload.get(protocol) or split_name) == split_name:
                protocol_groups[protocol].append(idx)

    def _protocol_acc(indices: List[int]) -> float:
        if not indices:
            return 0.0
        vals = [correctness[i] for i in indices if i < len(correctness)]
        return float(np.mean(vals)) if vals else 0.0
    def _protocol_macro_f1(indices: List[int]) -> float:
        if not indices:
            return 0.0
        true_vals = [y_true[i] for i in indices if i < len(y_true)]
        pred_vals = [y_pred[i] for i in indices if i < len(y_pred)]
        if not true_vals:
            return 0.0
        return float(f1_score(true_vals, pred_vals, average="macro"))
    metrics = {
        "split": split_name,
        "num_episodes": len(episodes),
        "num_queries": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "aspect_only_accuracy": float(accuracy_score(y_true_aspect, y_pred_aspect)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "flexible_match_score": flex_correct_rate,
        "multi_label_overlap_score": avg_overlap,
        "abstention_precision": float(abstain_precision),
        "abstention_recall": float(abstain_recall),
        "abstention_f1": float(abstain_f1),
        "coverage": float(coverage),
        "risk": float(risk),
        "coverage_risk": {"coverage": float(coverage), "risk": float(risk)},
        "known_vs_novel_quality": float(known_novel_quality),
        "known_vs_novel_auroc": float(known_vs_novel_auroc),
        "known_vs_novel_f1_macro": float(known_novel_f1_macro),
        "known_vs_novel_f1": {"known": float(f1_known), "novel": float(f1_novel)},
        "known_vs_novel_confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "known_vs_novel_not_applicable": bool(not has_novel_positives),
        "open_set_risk_coverage_curve": open_set_curve,
        "boundary_abstain_quality": float(boundary_abstain_quality),
        "novel_cluster_consistency": float(cluster_consistency),
        "ambiguity_sliced": {"high_ambiguity_accuracy": float(high_ambiguity_accuracy)},
        "protocol_breakdown": {
            "random": {"accuracy": _protocol_acc(protocol_groups["random"]), "macro_f1": _protocol_macro_f1(protocol_groups["random"])},
            "grouped": {"accuracy": _protocol_acc(protocol_groups["grouped"]), "macro_f1": _protocol_macro_f1(protocol_groups["grouped"])},
            "domain_holdout": {"accuracy": _protocol_acc(protocol_groups["domain_holdout"]), "macro_f1": _protocol_macro_f1(protocol_groups["domain_holdout"])},
        },
        "per_aspect_accuracy": {aspect: float(sum(values) / max(1, len(values))) for aspect, values in sorted(per_aspect.items())},
        "calibration_ece": _expected_calibration_error(confidences, correctness),
        "low_confidence_rate": low_confidence_count / max(1, len(y_true)),
        "inference_seconds": elapsed,
        "avg_seconds_per_episode": elapsed / max(1, len(episodes)),
    }
    joint_mode_metrics = _compact_mode_metrics(predictions, cfg, split_name, mode="joint", elapsed=elapsed, episodes=episodes)
    post_aspect_rows = _project_prediction_rows(predictions, "post_aspect")
    post_aspect_mode_metrics = _compact_mode_metrics(post_aspect_rows, cfg, split_name, mode="post_aspect", elapsed=elapsed, episodes=episodes)
    metrics["selected_mode"] = cfg.sentiment_pipeline
    metrics["primary_modes"] = ["joint", "post_aspect", "abstain_aware"]
    metrics["mode_metrics"] = {
        "joint": joint_mode_metrics,
        "post_aspect": post_aspect_mode_metrics,
    }
    if cfg.sentiment_pipeline == "post_aspect":
        metrics.update(post_aspect_mode_metrics)
    return metrics, predictions
