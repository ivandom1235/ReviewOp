from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from .dataset import SentenceDataset, diagnostics_to_dict, load_dataset_bundle
from .inference import ImplicitAspectDetector


def _progress_iter(iterable, total: int | None = None, desc: str = "", disable: bool = False):
    if disable:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


@dataclass(frozen=True)
class SentencePrediction:
    sentence: str
    true_labels: List[str]
    predicted_labels: List[str]
    ranked_scores: Dict[str, float]


def predict_aspects(
    sentence: str,
    top_k: int = 3,
    threshold: float = 0.6,
    prototypes_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str | None = None,
    return_top1_if_empty: bool = False,
) -> List[Dict[str, float | str]]:
    proto_path = (
        Path(prototypes_path)
        if prototypes_path
        else Path(__file__).resolve().parents[1] / "models" / "implicit_proto" / "prototypes.npz"
    )
    detector = ImplicitAspectDetector.from_artifacts(
        prototypes_path=proto_path,
        model_name=model_name,
        device=device,
    )
    return detector.predict_aspect_dicts(
        sentence=sentence,
        top_k=top_k,
        threshold=threshold,
        return_top1_if_empty=return_top1_if_empty,
    )


def _metric_block(
    y_true: List[Sequence[str]],
    y_pred: List[Sequence[str]],
    classes: List[str],
    top_k: int,
    threshold: float,
) -> Dict[str, object]:
    if not classes:
        return {
            "num_sentences": len(y_true),
            "num_classes": 0,
            "top_k": top_k,
            "threshold": threshold,
            "subset_accuracy": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_aspect": [],
        }

    mlb = MultiLabelBinarizer(classes=classes)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="micro",
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="macro",
        zero_division=0,
    )

    subset_acc = accuracy_score(y_true_bin, y_pred_bin)
    per_aspect_p, per_aspect_r, per_aspect_f1, support = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average=None,
        zero_division=0,
    )
    per_aspect = []
    for idx, label in enumerate(mlb.classes_):
        per_aspect.append(
            {
                "aspect": label,
                "precision": float(per_aspect_p[idx]),
                "recall": float(per_aspect_r[idx]),
                "f1": float(per_aspect_f1[idx]),
                "support": int(support[idx]),
            }
        )

    return {
        "num_sentences": len(y_true),
        "num_classes": len(classes),
        "top_k": top_k,
        "threshold": threshold,
        "subset_accuracy": float(subset_acc),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_aspect": per_aspect,
    }


def _evaluate_predictions(
    predictions: List[SentencePrediction],
    top_k: int,
    threshold: float,
    train_labels: Sequence[str] | None = None,
) -> Dict[str, object]:
    train_label_set = set(train_labels or [])
    full_classes = sorted({label for row in predictions for label in row.true_labels} | {label for row in predictions for label in row.predicted_labels})
    full_true = [row.true_labels for row in predictions]
    full_pred = [row.predicted_labels for row in predictions]
    full_metrics = _metric_block(full_true, full_pred, full_classes, top_k, threshold)

    seen_true = [[label for label in row.true_labels if label in train_label_set] for row in predictions]
    seen_pred = [[label for label in row.predicted_labels if label in train_label_set] for row in predictions]
    seen_classes = sorted(train_label_set & set(full_classes)) if train_label_set else full_classes
    seen_metrics = _metric_block(seen_true, seen_pred, seen_classes, top_k, threshold)

    prediction_sizes = Counter(len(row.predicted_labels) for row in predictions)
    no_prediction_rate = (
        sum(1 for row in predictions if not row.predicted_labels) / max(1, len(predictions))
    )
    true_label_set = sorted({label for row in predictions for label in row.true_labels})
    train_coverage = (
        (len(set(true_label_set) & train_label_set) / max(1, len(true_label_set))) * 100.0 if train_label_set else 100.0
    )

    summary = {
        "num_sentences": len(predictions),
        "num_classes": full_metrics["num_classes"],
        "top_k": top_k,
        "threshold": threshold,
        "subset_accuracy": float(full_metrics["subset_accuracy"]),
        "micro_precision": float(full_metrics["micro_precision"]),
        "micro_recall": float(full_metrics["micro_recall"]),
        "micro_f1": float(full_metrics["micro_f1"]),
        "macro_precision": float(full_metrics["macro_precision"]),
        "macro_recall": float(full_metrics["macro_recall"]),
        "macro_f1": float(full_metrics["macro_f1"]),
        "train_label_coverage": round(train_coverage, 4),
        "prediction_count_distribution": {str(k): int(v) for k, v in sorted(prediction_sizes.items())},
        "no_prediction_rate": round(float(no_prediction_rate), 6),
    }

    per_aspect = full_metrics["per_aspect"]
    confusion_pairs = Counter()
    for row in predictions:
        if row.true_labels and row.predicted_labels:
            for t in row.true_labels:
                for p in row.predicted_labels:
                    if t != p:
                        confusion_pairs[f"{t}->{p}"] += 1

    bins = {"1": [], "2_3": [], "4_5": [], "gt5": []}
    for item in per_aspect:
        support = int(item.get("support", 0))
        if support <= 1:
            bins["1"].append(float(item.get("f1", 0.0)))
        elif support <= 3:
            bins["2_3"].append(float(item.get("f1", 0.0)))
        elif support <= 5:
            bins["4_5"].append(float(item.get("f1", 0.0)))
        else:
            bins["gt5"].append(float(item.get("f1", 0.0)))

    def _avg(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    support_bin_macro_f1 = {k: _avg(v) for k, v in bins.items()}
    zero_recall_labels = sorted([str(item["aspect"]) for item in per_aspect if float(item.get("recall", 0.0)) == 0.0])
    zero_support_labels = sorted([str(item["aspect"]) for item in per_aspect if int(item.get("support", 0)) == 0])

    return {
        "summary": summary,
        "full_metrics": full_metrics,
        "train_label_metrics": seen_metrics,
        "per_aspect": full_metrics["per_aspect"],
        "diagnostics": {
            "support_bin_macro_f1": support_bin_macro_f1,
            "support_bin_counts": {k: len(v) for k, v in bins.items()},
            "zero_recall_labels": zero_recall_labels,
            "zero_support_labels": zero_support_labels,
            "top_confusions": [
                {"pair": pair, "count": int(cnt)}
                for pair, cnt in confusion_pairs.most_common(20)
            ],
        },
    }


def evaluate_dataset(
    detector: ImplicitAspectDetector,
    dataset: SentenceDataset,
    top_k: int = 3,
    threshold: float = 0.6,
    return_top1_if_empty: bool = False,
    show_progress: bool = True,
    train_labels: Sequence[str] | None = None,
    label_thresholds: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    sentence_to_true = dataset.unique_sentences_with_labels()
    predictions: List[SentencePrediction] = []

    iterator = _progress_iter(
        sentence_to_true.items(),
        total=len(sentence_to_true),
        desc="Evaluating sentences",
        disable=not show_progress,
    )
    for sentence, true_labels in iterator:
        score_map = detector.score_sentence(sentence)
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        predicted_labels = []
        for label, score in ranked[: max(1, top_k)]:
            effective_threshold = float(label_thresholds.get(label, threshold)) if label_thresholds else float(threshold)
            if score >= effective_threshold:
                predicted_labels.append(label)
        if not predicted_labels and return_top1_if_empty and ranked:
            predicted_labels = [ranked[0][0]]
        predictions.append(
            SentencePrediction(
                sentence=sentence,
                true_labels=list(true_labels),
                predicted_labels=predicted_labels,
                ranked_scores={label: round(float(score), 6) for label, score in ranked},
            )
        )

    report = _evaluate_predictions(
        predictions=predictions,
        top_k=top_k,
        threshold=threshold,
        train_labels=train_labels,
    )
    report["predictions"] = [asdict(row) for row in predictions]
    if label_thresholds:
        report["label_thresholds"] = {label: float(value) for label, value in sorted(label_thresholds.items())}
    return report


def calibrate_label_thresholds(
    detector: ImplicitAspectDetector,
    dataset: SentenceDataset,
    candidate_thresholds: Sequence[float],
    train_labels: Sequence[str],
    top_k: int,
    base_threshold: float,
    min_support: int = 3,
    min_threshold: float = 0.35,
    max_threshold: float = 0.6,
) -> Dict[str, float]:
    sentence_to_true = dataset.unique_sentences_with_labels()
    if not sentence_to_true:
        return {}

    score_rows = [(sentence, set(labels), detector.score_sentence(sentence)) for sentence, labels in sentence_to_true.items()]
    label_thresholds: Dict[str, float] = {}
    train_label_set = set(train_labels)

    for label in sorted(train_label_set):
        support = sum(1 for _, labels, _ in score_rows if label in labels)
        if support < min_support:
            continue

        best_threshold = float(base_threshold)
        best_f1 = -1.0
        for candidate in candidate_thresholds:
            y_true: List[List[str]] = []
            y_pred: List[List[str]] = []
            for _, labels, score_map in score_rows:
                ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
                preds = [name for name, score in ranked[: max(1, top_k)] if score >= candidate and name == label]
                y_true.append([label] if label in labels else [])
                y_pred.append(preds)
            metrics = _metric_block(y_true, y_pred, [label], top_k, float(candidate))
            f1 = float(metrics["macro_f1"])
            if f1 > best_f1 or (f1 == best_f1 and candidate < best_threshold):
                best_f1 = f1
                best_threshold = float(candidate)
        label_thresholds[label] = min(max(float(best_threshold), float(min_threshold)), float(max_threshold))
    return label_thresholds


def evaluate_default_backend_split(
    split: str,
    prototypes_path: str | Path,
    top_k: int = 3,
    threshold: float = 0.6,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str | None = None,
    backend_root: str | Path | None = None,
    dataset_family: str = "reviewlevel",
    input_dir: str | Path | None = None,
    data_source: str = "backend_raw",
    show_progress: bool = True,
    return_top1_if_empty: bool = False,
    train_labels: Sequence[str] | None = None,
    label_thresholds: Mapping[str, float] | None = None,
    label_merge_enabled: bool = True,
    label_merge_map: Mapping[str, str] | None = None,
    label_merge_config: str | Path | None = None,
) -> Dict[str, object]:
    bundle = load_dataset_bundle(
        dataset_family=dataset_family,
        data_source=data_source,
        input_dir=input_dir,
        root_dir=backend_root,
        label_merge_enabled=label_merge_enabled,
        label_merge_map=label_merge_map,
        label_merge_config=label_merge_config,
    )
    dataset = bundle.splits[split]
    detector = ImplicitAspectDetector.from_artifacts(
        prototypes_path=prototypes_path,
        model_name=model_name,
        device=device,
    )
    report = evaluate_dataset(
        detector=detector,
        dataset=dataset,
        top_k=top_k,
        threshold=threshold,
        return_top1_if_empty=return_top1_if_empty,
        show_progress=show_progress,
        train_labels=train_labels or bundle.diagnostics.train_labels,
        label_thresholds=label_thresholds,
    )
    report["dataset_diagnostics"] = diagnostics_to_dict(bundle.diagnostics)
    report["split"] = split
    report["data_source"] = bundle.data_source
    return report
