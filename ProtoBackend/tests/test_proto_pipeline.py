from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "ProtoBackend") not in sys.path:
    sys.path.insert(0, str(ROOT / "ProtoBackend"))

from implicit_proto.dataset import SentenceDataset, SentenceRow, load_dataset_bundle, resolve_split_path
from implicit_proto.prototype_builder import PrototypeBuilder
from implicit_proto.test import calibrate_label_thresholds, evaluate_dataset


class FakeEncoder:
    def encode(self, sentences, batch_size: int = 32, normalize_embeddings: bool = True):
        rows = []
        for idx, sentence in enumerate(sentences):
            base = float(len(sentence))
            vector = np.array([base, float(idx + 1), float((idx % 3) + 1)], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


class DummyPrediction:
    def __init__(self, aspect: str):
        self.aspect = aspect


class DummyDetector:
    def __init__(self, score_table):
        self.score_table = score_table
        self.labels = sorted({label for scores in score_table.values() for label in scores})

    def score_sentence(self, sentence: str):
        return dict(self.score_table[sentence])

    def predict_aspects(self, sentence: str, top_k: int, threshold: float, return_top1_if_empty: bool = False, label_thresholds=None):
        label_thresholds = label_thresholds or {}
        ranked = sorted(self.score_table[sentence].items(), key=lambda x: x[1], reverse=True)
        kept = []
        for label, score in ranked[:top_k]:
            if score >= label_thresholds.get(label, threshold):
                kept.append(DummyPrediction(label))
        if not kept and return_top1_if_empty and ranked:
            kept.append(DummyPrediction(ranked[0][0]))
        return kept


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ProtoPipelineTests(unittest.TestCase):
    def test_resolve_split_path_prefers_backend_raw(self):
        path = resolve_split_path("train", dataset_family="episodic", data_source="backend_raw", root_dir=ROOT)
        self.assertEqual(path, ROOT / "backend" / "data" / "implicit" / "raw" / "implicit_episode_train.jsonl")

    def test_load_dataset_bundle_from_input_dir_detects_degenerate_validation(self):
        base = ROOT / "ProtoBackend" / "tests" / "_tmp_input_case"
        if base.exists():
            for path in sorted(base.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        _write_jsonl(
            base / "episodic" / "train.jsonl",
            [
                {"episode_id": "a", "evidence_sentence": "fast battery", "implicit_aspect": "battery"},
                {"episode_id": "b", "evidence_sentence": "slow network", "implicit_aspect": "network"},
            ],
        )
        _write_jsonl(
            base / "episodic" / "val.jsonl",
            [{"episode_id": "c", "evidence_sentence": "rude waiter", "implicit_aspect": "service"}],
        )
        _write_jsonl(
            base / "episodic" / "test.jsonl",
            [{"episode_id": "d", "evidence_sentence": "bad battery", "implicit_aspect": "battery"}],
        )

        try:
            bundle = load_dataset_bundle(
                dataset_family="episodic",
                data_source="input_dir",
                input_dir=base,
                root_dir=ROOT,
            )

            self.assertTrue(bundle.diagnostics.is_degenerate_validation)
            self.assertIn("service", bundle.diagnostics.split_stats["val"].unseen_labels_vs_train)
        finally:
            if base.exists():
                for path in sorted(base.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()

    def test_label_merge_map_applies_deterministically(self):
        base = ROOT / "ProtoBackend" / "tests" / "_tmp_input_case"
        _write_jsonl(
            base / "episodic" / "train.jsonl",
            [
                {"episode_id": "a", "evidence_sentence": "slow signal", "implicit_aspect": "signal_coverage"},
                {"episode_id": "b", "evidence_sentence": "bad network", "implicit_aspect": "network"},
            ],
        )
        _write_jsonl(base / "episodic" / "val.jsonl", [{"episode_id": "c", "evidence_sentence": "weak bars", "implicit_aspect": "signal_coverage"}])
        _write_jsonl(base / "episodic" / "test.jsonl", [{"episode_id": "d", "evidence_sentence": "lost calls", "implicit_aspect": "signal_coverage"}])

        try:
            bundle = load_dataset_bundle(
                dataset_family="episodic",
                data_source="input_dir",
                input_dir=base,
                root_dir=ROOT,
                label_merge_enabled=True,
                label_merge_map={"signal_coverage": "network"},
            )
            self.assertIn("network", bundle.splits["train"].unique_aspects())
            self.assertNotIn("signal_coverage", bundle.splits["train"].unique_aspects())
            self.assertGreaterEqual(sum(bundle.diagnostics.merged_label_counts.values()), 1)
        finally:
            if base.exists():
                for path in sorted(base.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()

    def test_prototype_builder_dedupes_and_adds_multiple_centroids(self):
        dataset = SentenceDataset(
            [
                SentenceRow("1", "alpha one", "battery", "train"),
                SentenceRow("2", "alpha one", "battery", "train"),
                SentenceRow("3", "alpha two", "battery", "train"),
                SentenceRow("4", "alpha three", "battery", "train"),
                SentenceRow("5", "alpha four", "battery", "train"),
                SentenceRow("6", "beta one", "screen", "train"),
                SentenceRow("7", "beta two", "screen", "train"),
            ]
        )
        builder = PrototypeBuilder(
            encoder=FakeEncoder(),
            dedupe_sentences=True,
            min_examples_per_centroid=2,
            max_centroids_per_label=2,
            shrinkage_alpha=1.0,
        )
        artifacts = builder.build(dataset, batch_size=2)

        self.assertEqual(artifacts.label_counts["battery"], 4)
        self.assertGreaterEqual(artifacts.prototypes.shape[0], 2)
        self.assertGreaterEqual(artifacts.centroid_labels.count("battery"), 1)

    def test_evaluate_dataset_reports_seen_label_metrics_for_unseen_validation_labels(self):
        dataset = SentenceDataset(
            [
                SentenceRow("1", "sent-a", "battery", "val"),
                SentenceRow("2", "sent-b", "service", "val"),
            ]
        )
        detector = DummyDetector(
            {
                "sent-a": {"battery": 0.8, "network": 0.3},
                "sent-b": {"battery": 0.7, "network": 0.2},
            }
        )

        report = evaluate_dataset(
            detector=detector,
            dataset=dataset,
            top_k=1,
            threshold=0.5,
            train_labels=["battery", "network"],
            show_progress=False,
        )

        self.assertEqual(report["full_metrics"]["num_classes"], 2)
        self.assertEqual(report["train_label_metrics"]["num_classes"], 1)
        self.assertEqual(report["summary"]["train_label_coverage"], 50.0)

    def test_calibrate_label_thresholds_skips_low_support_labels(self):
        dataset = SentenceDataset(
            [
                SentenceRow("1", "s1", "battery", "val"),
                SentenceRow("2", "s2", "battery", "val"),
                SentenceRow("3", "s3", "battery", "val"),
                SentenceRow("4", "s4", "network", "val"),
            ]
        )
        detector = DummyDetector(
            {
                "s1": {"battery": 0.8, "network": 0.1},
                "s2": {"battery": 0.75, "network": 0.2},
                "s3": {"battery": 0.9, "network": 0.1},
                "s4": {"battery": 0.3, "network": 0.7},
            }
        )

        thresholds = calibrate_label_thresholds(
            detector=detector,
            dataset=dataset,
            candidate_thresholds=[0.4, 0.5, 0.6],
            train_labels=["battery", "network"],
            top_k=1,
            base_threshold=0.5,
            min_support=3,
            min_threshold=0.45,
            max_threshold=0.55,
        )

        self.assertIn("battery", thresholds)
        self.assertNotIn("network", thresholds)
        self.assertGreaterEqual(thresholds["battery"], 0.45)
        self.assertLessEqual(thresholds["battery"], 0.55)


if __name__ == "__main__":
    unittest.main()
