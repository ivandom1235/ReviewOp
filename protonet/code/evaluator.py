from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

try:
    from .config import ProtonetConfig
    from .progress import task_bar
except ImportError:
    from config import ProtonetConfig
    from progress import task_bar


def _aspect_from_joint(label: str) -> str:
    return label.split("__", 1)[0]


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
    per_aspect: Dict[str, List[int]] = defaultdict(list)
    predictions: List[Dict[str, Any]] = []

    start_time = time.perf_counter()
    with torch.no_grad():
        with task_bar(total=len(episodes), desc=f"eval:{split_name}", enabled=cfg.progress_enabled) as bar:
            for episode in episodes:
                out = model.episode_forward(episode)
                probs = out.probabilities.numpy()
                target_indices = out.targets.detach().cpu().tolist()
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
                    y_true_aspect.append(_aspect_from_joint(true_label))
                    y_pred_aspect.append(_aspect_from_joint(pred_label))
                    confidences.append(confidence)
                    is_correct = int(true_label == pred_label)
                    correctness.append(is_correct)
                    if confidence < cfg.low_confidence_threshold:
                        low_confidence_count += 1
                    per_aspect[_aspect_from_joint(true_label)].append(is_correct)
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
                        }
                    )
                bar.update(1)
                if y_true:
                    bar.set_postfix(acc=f"{accuracy_score(y_true, y_pred):.3f}")

    elapsed = time.perf_counter() - start_time
    metrics = {
        "split": split_name,
        "num_episodes": len(episodes),
        "num_queries": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "aspect_only_accuracy": float(accuracy_score(y_true_aspect, y_pred_aspect)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0,
        "per_aspect_accuracy": {aspect: float(sum(values) / max(1, len(values))) for aspect, values in sorted(per_aspect.items())},
        "calibration_ece": _expected_calibration_error(confidences, correctness),
        "low_confidence_rate": low_confidence_count / max(1, len(y_true)),
        "inference_seconds": elapsed,
        "avg_seconds_per_episode": elapsed / max(1, len(episodes)),
    }
    return metrics, predictions
