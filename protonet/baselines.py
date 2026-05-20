from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import DatasetBundle
from .encoder import build_encoder
from .label_normalizer import LabelNormalizer


@dataclass
class BaselineOutput:
    y_true_sets: list[set[str]]
    y_pred_sets: list[set[str]]
    y_true_unknown: list[int]
    y_score_unknown: list[float]
    y_true_matrix: np.ndarray
    y_pred_matrix: np.ndarray


def _gold_sets(rows, normalizer: LabelNormalizer) -> list[set[str]]:
    out: list[set[str]] = []
    for ex in rows:
        out.append(normalizer.normalize_set([g.aspect for g in ex.gold_aspects if g.aspect]))
    return out


def _label_space(train_sets: list[set[str]]) -> list[str]:
    labels = sorted({x for s in train_sets for x in s if x and x != "unknown"})
    return labels


def _sets_to_matrix(sets_: list[set[str]], labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(sets_), len(labels)), dtype=np.int32)
    for r, s in enumerate(sets_):
        for l in s:
            if l in idx:
                m[r, idx[l]] = 1
    return m


def run_logreg_baseline(
    bundle: DatasetBundle,
    split_rows,
    *,
    train_rows=None,
    encoder_kind: str = "hashing",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    threshold: float = 0.5,
) -> BaselineOutput:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_rows = train_rows if train_rows is not None else bundle.train
    train_sets = _gold_sets(train_rows, bundle.normalizer)
    eval_sets = _gold_sets(split_rows, bundle.normalizer)
    labels = _label_space(train_sets)
    y_train = _sets_to_matrix(train_sets, labels)
    y_eval = _sets_to_matrix(eval_sets, labels)

    encoder = build_encoder(encoder_kind, model_name, True, 2048, batch_size)
    x_train = encoder.encode([ex.text for ex in train_rows])
    x_eval = encoder.encode([ex.text for ex in split_rows])

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_eval)
    pred = (prob >= threshold).astype(np.int32)

    y_pred_sets: list[set[str]] = []
    for row in pred:
        y_pred_sets.append({labels[i] for i, v in enumerate(row.tolist()) if v == 1})

    y_true_unknown = [1 if ex.gold_unseen_labels else 0 for ex in split_rows]
    max_prob = prob.max(axis=1) if prob.size else np.zeros((len(split_rows),), dtype=np.float32)
    y_score_unknown = (1.0 - max_prob).tolist()

    return BaselineOutput(
        y_true_sets=eval_sets,
        y_pred_sets=y_pred_sets,
        y_true_unknown=y_true_unknown,
        y_score_unknown=y_score_unknown,
        y_true_matrix=y_eval,
        y_pred_matrix=pred,
    )


def run_centroid_baseline(
    bundle: DatasetBundle,
    split_rows,
    *,
    train_rows=None,
    encoder_kind: str = "hashing",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    accept_threshold: float = 0.30,
) -> BaselineOutput:
    train_rows = train_rows if train_rows is not None else bundle.train
    train_sets = _gold_sets(train_rows, bundle.normalizer)
    eval_sets = _gold_sets(split_rows, bundle.normalizer)
    labels = _label_space(train_sets)
    y_eval = _sets_to_matrix(eval_sets, labels)

    encoder = build_encoder(encoder_kind, model_name, True, 2048, batch_size)
    x_train = encoder.encode([ex.text for ex in train_rows])
    x_eval = encoder.encode([ex.text for ex in split_rows])

    idx = {l: i for i, l in enumerate(labels)}
    sums = np.zeros((len(labels), x_train.shape[1]), dtype=np.float32)
    counts = np.zeros((len(labels),), dtype=np.float32)
    for r, s in enumerate(train_sets):
        for l in s:
            if l in idx:
                sums[idx[l]] += x_train[r]
                counts[idx[l]] += 1.0
    counts = np.maximum(counts, 1.0)
    centroids = sums / counts[:, None]
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = np.divide(centroids, np.maximum(norms, 1e-8))

    sim = np.matmul(x_eval.astype(np.float32), centroids.astype(np.float32).T) if centroids.size else np.zeros((len(split_rows), 0), dtype=np.float32)
    max_sim = sim.max(axis=1) if sim.size else np.zeros((len(split_rows),), dtype=np.float32)
    top_idx = sim.argmax(axis=1) if sim.size else np.zeros((len(split_rows),), dtype=np.int32)

    y_pred_sets: list[set[str]] = []
    for i in range(len(split_rows)):
        if sim.size and float(max_sim[i]) >= accept_threshold:
            y_pred_sets.append({labels[int(top_idx[i])]})
        else:
            y_pred_sets.append(set())

    y_pred = _sets_to_matrix(y_pred_sets, labels)
    y_true_unknown = [1 if ex.gold_unseen_labels else 0 for ex in split_rows]
    y_score_unknown = (1.0 - max_sim).tolist()

    return BaselineOutput(
        y_true_sets=eval_sets,
        y_pred_sets=y_pred_sets,
        y_true_unknown=y_true_unknown,
        y_score_unknown=y_score_unknown,
        y_true_matrix=y_eval,
        y_pred_matrix=y_pred,
    )
