from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

try:
    from .config import ProtonetConfig
    from .progress import task_bar
except ImportError:
    from config import ProtonetConfig
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


def evaluate_episodes(model, episodes: List[Dict[str, Any]], cfg: ProtonetConfig, split_name: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
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
                    gold_joint_labels = list(query_row.get("gold_joint_labels") or []) if isinstance(query_row, dict) else []
                    true_set = set(gold_joint_labels) if gold_joint_labels else {true_label}
                    pred_set = {pred_label}
                    jaccard = len(true_set & pred_set) / max(1, len(true_set | pred_set))
                    novelty_score = 1.0 - confidence
                    split_protocol = query_row.get("split_protocol") if isinstance(query_row, dict) else {}
                    novelty_truth_label = 1 if bool(query_row.get("novel_aspect_acceptable", False)) else 0
                    novelty_truth.append(novelty_truth_label)
                    novelty_scores.append(novelty_score)
                    predictions.append(
                        {
                            "episode_id": episode.get("episode_id"),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "confidence": confidence,
                            "low_confidence": confidence < cfg.low_confidence_threshold,
                            "top_k": top_predictions,
                            "correct": bool(is_correct),
                            "split": split_name,
                            "flex_correct": bool(pred_label in true_set),
                            "multi_label_overlap": float(jaccard),
                            "abstained": did_abstain,
                            "novelty_score": float(novelty_score),
                            "routing": "novel" if novelty_score >= cfg.novelty_threshold else "known",
                            "split_protocol": split_protocol if isinstance(split_protocol, dict) else {},
                            "benchmark_ambiguity_score": float(query_row.get("benchmark_ambiguity_score", 0.0)) if isinstance(query_row, dict) else 0.0,
                            "abstain_acceptable": bool(query_row.get("abstain_acceptable", False)) if isinstance(query_row, dict) else False,
                            "novel_aspect_acceptable": bool(query_row.get("novel_aspect_acceptable", False)) if isinstance(query_row, dict) else False,
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
    known_novel_quality = float(np.mean([1.0 - abs(float(row.get("novelty_score", 0.0)) - float(row.get("novel_aspect_acceptable", 0.0))) for row in predictions])) if predictions else 0.0
    known_vs_novel_auroc = 0.0
    if len(set(novelty_truth)) > 1:
        try:
            known_vs_novel_auroc = float(roc_auc_score(novelty_truth, novelty_scores))
        except ValueError:
            known_vs_novel_auroc = 0.0
    high_ambiguity_values = [1.0 if row.get("correct") else 0.0 for row in predictions if float(row.get("benchmark_ambiguity_score", 0.0)) >= 0.5]
    high_ambiguity_accuracy = float(np.mean(high_ambiguity_values)) if high_ambiguity_values else 0.0
    protocol_groups: Dict[str, List[int]] = {"random": [], "source_holdout": [], "domain_holdout": []}
    for idx, row in enumerate(predictions):
        protocol_payload = row.get("split_protocol") or {}
        for protocol in ("random", "source_holdout", "domain_holdout"):
            if str(protocol_payload.get(protocol) or split_name) == split_name:
                protocol_groups[protocol].append(idx)

    def _protocol_acc(indices: List[int]) -> float:
        if not indices:
            return 0.0
        vals = [correctness[i] for i in indices if i < len(correctness)]
        return float(np.mean(vals)) if vals else 0.0
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
        "ambiguity_sliced": {"high_ambiguity_accuracy": float(high_ambiguity_accuracy)},
        "protocol_breakdown": {
            "random": {"accuracy": _protocol_acc(protocol_groups["random"])},
            "source_holdout": {"accuracy": _protocol_acc(protocol_groups["source_holdout"])},
            "domain_holdout": {"accuracy": _protocol_acc(protocol_groups["domain_holdout"])},
        },
        "per_aspect_accuracy": {aspect: float(sum(values) / max(1, len(values))) for aspect, values in sorted(per_aspect.items())},
        "calibration_ece": _expected_calibration_error(confidences, correctness),
        "low_confidence_rate": low_confidence_count / max(1, len(y_true)),
        "inference_seconds": elapsed,
        "avg_seconds_per_episode": elapsed / max(1, len(episodes)),
    }
    return metrics, predictions
