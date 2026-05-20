from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def micro_f1_from_sets(y_true: list[set[str]], y_pred: list[set[str]]) -> float:
    tp = fp = fn = 0
    for gold, pred in zip(y_true, y_pred):
        inter = len(gold & pred)
        tp += inter
        fp += max(0, len(pred) - inter)
        fn += max(0, len(gold) - inter)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r)) if p + r else 0.0


def macro_f1_from_binary_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import f1_score

        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return 0.0


def bootstrap_ci(
    metric_fn: Callable[[list[set[str]], list[set[str]]], float],
    y_true: list[set[str]],
    y_pred: list[set[str]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt_true = [y_true[i] for i in idx]
        bt_pred = [y_pred[i] for i in idx]
        vals.append(metric_fn(bt_true, bt_pred))
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(arr, 0.025)),
        "upper": float(np.quantile(arr, 0.975)),
    }


def paired_bootstrap_delta(
    metric_fn: Callable[[list[set[str]], list[set[str]]], float],
    y_true: list[set[str]],
    pred_a: list[set[str]],
    pred_b: list[set[str]],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"delta_mean": 0.0, "p_value": 1.0}
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt_true = [y_true[i] for i in idx]
        bt_a = [pred_a[i] for i in idx]
        bt_b = [pred_b[i] for i in idx]
        deltas.append(metric_fn(bt_true, bt_a) - metric_fn(bt_true, bt_b))
    arr = np.array(deltas, dtype=np.float64)
    p_two_sided = 2.0 * min(float((arr <= 0).mean()), float((arr >= 0).mean()))
    return {"delta_mean": float(arr.mean()), "p_value": min(1.0, p_two_sided)}

