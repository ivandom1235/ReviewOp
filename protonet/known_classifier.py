from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import TextEncoder
from .label_normalizer import LabelNormalizer


@dataclass
class KnownClassifierModel:
    labels: list[str]
    model: object
    encoder: TextEncoder

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        x = self.encoder.encode(texts)
        return self.model.predict_proba(x)


def _gold_sets(rows, normalizer: LabelNormalizer) -> list[set[str]]:
    out: list[set[str]] = []
    for ex in rows:
        out.append(normalizer.normalize_set([g.aspect for g in ex.gold_aspects if g.aspect and g.aspect != "unknown"]))
    return out


def build_known_classifier(
    train_rows,
    normalizer: LabelNormalizer,
    encoder: TextEncoder,
    *,
    max_labels: int = 50,
) -> KnownClassifierModel | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_sets = _gold_sets(train_rows, normalizer)
    labels = sorted({x for s in train_sets for x in s if x and x != "unknown"})
    if not labels:
        return None
    if len(labels) > int(max_labels):
        return None
    idx = {l: i for i, l in enumerate(labels)}
    y = np.zeros((len(train_sets), len(labels)), dtype=np.int32)
    for r, s in enumerate(train_sets):
        for l in s:
            if l in idx:
                y[r, idx[l]] = 1
    if y.sum() == 0:
        return None
    if not hasattr(encoder, "encode"):
        return None
    x = encoder.encode([ex.text for ex in train_rows])
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        ),
        n_jobs=1,
    )
    clf.fit(x, y)
    return KnownClassifierModel(labels=labels, model=clf, encoder=encoder)
