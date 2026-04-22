from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class BenchmarkDedupTests(unittest.TestCase):
    def test_logical_row_key_ignores_domain_and_group_when_text_matches(self) -> None:
        from build_dataset import _benchmark_logical_row_key

        row_a = {
            "domain": "telecom",
            "group_id": "g-telecom",
            "source_text": "The fan noise was so loud it drowned out the video.",
            "gold_interpretations": [
                {"aspect_label": "noise", "sentiment": "negative", "evidence_text": "fan noise"},
            ],
        }
        row_b = {
            "domain": "healthcare",
            "group_id": "g-healthcare",
            "source_text": "The fan noise was so loud it drowned out the video.",
            "gold_interpretations": [
                {"aspect_label": "noise", "sentiment": "negative", "evidence_text": "fan noise"},
            ],
        }

        self.assertEqual(
            _benchmark_logical_row_key(row_a["source_text"], row_a["gold_interpretations"]),
            _benchmark_logical_row_key(row_b["source_text"], row_b["gold_interpretations"]),
        )

    def test_logical_row_key_changes_when_interpretation_changes(self) -> None:
        from build_dataset import _benchmark_logical_row_key

        row_a = {
            "source_text": "The fan noise was so loud it drowned out the video.",
            "gold_interpretations": [
                {"aspect_label": "noise", "sentiment": "negative", "evidence_text": "fan noise"},
            ],
        }
        row_b = {
            "source_text": "The fan noise was so loud it drowned out the video.",
            "gold_interpretations": [
                {"aspect_label": "video", "sentiment": "negative", "evidence_text": "video"},
            ],
        }

        self.assertNotEqual(
            _benchmark_logical_row_key(row_a["source_text"], row_a["gold_interpretations"]),
            _benchmark_logical_row_key(row_b["source_text"], row_b["gold_interpretations"]),
        )

    def test_canonicalized_aspect_labels_collapse_fragmented_restaurant_rows(self) -> None:
        from build_dataset import _normalize_interpretation_contract

        row = {"domain": "restaurant"}
        registry = {"registry_version": "v1", "domains": {}}

        wait_staff = _normalize_interpretation_contract(
            interpretation={
                "aspect_label": "wait staff",
                "sentiment": "negative",
                "evidence_text": "wait staff",
                "label_type": "implicit",
                "source": "synthetic",
            },
            domain="restaurant",
            registry=registry,
            enforce_registry_membership=False,
        )
        meals = _normalize_interpretation_contract(
            interpretation={
                "aspect_label": "meals",
                "sentiment": "negative",
                "evidence_text": "meals",
                "label_type": "implicit",
                "source": "synthetic",
            },
            domain="restaurant",
            registry=registry,
            enforce_registry_membership=False,
        )

        self.assertIsNotNone(wait_staff)
        self.assertIsNotNone(meals)
        self.assertEqual(wait_staff["domain_canonical_aspect"], "service_speed")
        self.assertEqual(meals["domain_canonical_aspect"], "food_quality")
        self.assertEqual(wait_staff["aspect_label"], "service_speed")
        self.assertEqual(meals["aspect_label"], "food_quality")
        self.assertIn("canonical_mapping_source", wait_staff)
        self.assertIn("canonical_mapping_confidence", wait_staff)
        self.assertGreaterEqual(float(wait_staff["canonical_mapping_confidence"]), 0.85)

    def test_normalize_interpretation_contract_rejects_generic_aspect(self) -> None:
        from build_dataset import _normalize_interpretation_contract

        normalized = _normalize_interpretation_contract(
            interpretation={
                "aspect_label": "quality",
                "sentiment": "negative",
                "evidence_text": "quality was bad",
                "label_type": "implicit",
                "source": "synthetic",
            },
            domain="restaurant",
            registry={"registry_version": "v1", "domains": {}},
            enforce_registry_membership=False,
        )

        self.assertIsNone(normalized)

    def test_cross_mode_duplicates_drop_explicit_copy_when_implicit_exists(self) -> None:
        from build_dataset import _collapse_cross_mode_gold_interpretations

        items = [
            {
                "aspect_label": "service",
                "domain_canonical_aspect": "service_speed",
                "sentiment": "negative",
                "evidence_text": "service was slow",
                "evidence_mode": "implicit",
            },
            {
                "aspect_label": "service",
                "domain_canonical_aspect": "service_speed",
                "sentiment": "negative",
                "evidence_text": "service was slow",
                "evidence_mode": "explicit",
            },
            {
                "aspect_label": "food",
                "domain_canonical_aspect": "food_quality",
                "sentiment": "positive",
                "evidence_text": "food was great",
                "evidence_mode": "explicit",
            },
        ]

        collapsed = _collapse_cross_mode_gold_interpretations(items)

        self.assertEqual(len(collapsed), 2)
        self.assertTrue(any(item["domain_canonical_aspect"] == "service_speed" for item in collapsed))
        self.assertTrue(any(item["domain_canonical_aspect"] == "food_quality" for item in collapsed))

    def test_preselect_benchmark_rows_keeps_one_logical_representative_and_preserves_family_fallback(self) -> None:
        from build_dataset import _preselect_benchmark_rows

        rows = [
            {
                "id": "telecom-1",
                "domain": "telecom",
                "review_text": "The signal was reliable all day.",
                "gold_interpretations": [
                    {"aspect_label": "signal", "sentiment": "positive", "evidence_text": "signal"},
                ],
                "implicit_grounded_interpretations": [{"aspect_label": "signal"}],
                "explicit_grounded_interpretations": [],
                "hardness_tier": "H1",
                "abstain_acceptable": False,
                "novel_acceptable": False,
            },
            {
                "id": "electronics-1",
                "domain": "electronics",
                "review_text": "The signal was reliable all day.",
                "gold_interpretations": [
                    {"aspect_label": "signal", "sentiment": "positive", "evidence_text": "signal"},
                ],
                "implicit_grounded_interpretations": [{"aspect_label": "signal"}, {"aspect_label": "signal"}],
                "explicit_grounded_interpretations": [],
                "hardness_tier": "H3",
                "abstain_acceptable": True,
                "novel_acceptable": True,
            },
        ]

        selected_rows, fallback_rows_by_family, stats = _preselect_benchmark_rows(rows, seed=7)

        self.assertEqual(len(selected_rows), 1)
        self.assertEqual(stats["logical_duplicates_removed"], 1)
        self.assertIn("telecom", fallback_rows_by_family)
        self.assertEqual(fallback_rows_by_family["telecom"][0]["domain"], "telecom")

    def test_benchmark_export_rejects_placeholder_review_text(self) -> None:
        from build_dataset import _build_benchmark_instances

        rows = [
            {
                "id": "placeholder-1",
                "domain": "legal",
                "review_text": "Generic positive review about reliability in legal domain sample 11.",
                "source_text": "Generic positive review about reliability in legal domain sample 11.",
                "gold_interpretations": [
                    {
                        "aspect_label": "reliability",
                        "sentiment": "positive",
                        "evidence_text": "Generic positive review about reliability in legal domain sample 11.",
                        "annotator_support": 1,
                        "source": "synthetic",
                        "label_type": "implicit",
                    }
                ],
                "implicit": {
                    "aspects": ["reliability"],
                    "spans": [
                        {
                            "latent_label": "reliability",
                            "evidence_text": "Generic positive review about reliability in legal domain sample 11.",
                            "support_type": "synthetic",
                            "confidence": 0.99,
                        }
                    ],
                },
                "split": "train",
                "group_id": "legal_gen_002",
                "abstain_acceptable": False,
                "novel_acceptable": False,
            }
        ]
        assignments = {
            "placeholder-1": {
                "random": "train",
                "grouped": "train",
                "domain_holdout": "train",
            }
        }

        benchmark_rows_by_split, metadata, deferred_rows = _build_benchmark_instances(
            rows,
            assignments,
            enforce_registry_membership=False,
        )

        self.assertEqual(benchmark_rows_by_split, {"train": [], "val": [], "test": []})
        self.assertEqual(len(deferred_rows), 1)
        self.assertEqual(deferred_rows[0]["reason"], "placeholder_review_text")
        self.assertEqual(metadata["split_counts"]["train"], 0)

    def test_benchmark_export_preserves_hardness_from_source_row(self) -> None:
        from build_dataset import _build_benchmark_instances

        rows = [
            {
                "id": "hardness-1",
                "domain": "electronics",
                "review_text": "The battery lasted all day without heating up.",
                "source_text": "The battery lasted all day without heating up.",
                "gold_interpretations": [
                    {
                        "aspect_label": "power",
                        "sentiment": "positive",
                        "evidence_text": "battery lasted all day",
                        "annotator_support": 2,
                        "source": "synthetic",
                        "label_type": "implicit",
                        "hardness_tier": "H3",
                    }
                ],
                "implicit": {
                    "aspects": ["power"],
                    "spans": [
                        {
                            "latent_label": "power",
                            "evidence_text": "battery lasted all day",
                            "support_type": "gold",
                            "confidence": 0.98,
                        }
                    ],
                },
                "split": "train",
                "group_id": "electronics_h3_001",
                "abstain_acceptable": True,
                "novel_acceptable": True,
            }
        ]
        assignments = {
            "hardness-1": {
                "random": "train",
                "grouped": "train",
                "domain_holdout": "train",
            }
        }

        benchmark_rows_by_split, metadata, deferred_rows = _build_benchmark_instances(
            rows,
            assignments,
            enforce_registry_membership=False,
        )

        self.assertEqual(len(deferred_rows), 0)
        self.assertEqual(len(benchmark_rows_by_split["val"]), 1)
        self.assertEqual(benchmark_rows_by_split["val"][0]["hardness_tier"], "H3")
        self.assertEqual(metadata["hardness_distribution"]["H3"], 1)

    def test_build_gold_interpretations_populates_spans_from_source_text(self) -> None:
        from build_dataset import _build_gold_interpretations, _sanitize_gold_interpretation_spans

        row = {
            "source_text": "The connection kept dropping until I restarted the modem.",
            "gold_interpretations": [
                {
                    "aspect": "connectivity",
                    "sentiment": "negative",
                    "evidence_text": "connection kept dropping",
                }
            ],
        }

        interpretations = _build_gold_interpretations(row)
        self.assertEqual(len(interpretations), 1)
        self.assertNotEqual(interpretations[0]["evidence_span"], [-1, -1])
        _, repaired, hard_failures = _sanitize_gold_interpretation_spans(row["source_text"], interpretations)
        self.assertEqual(repaired, 0)
        self.assertEqual(hard_failures, 0)

    def test_placeholder_review_text_is_not_strict_train_eligible(self) -> None:
        from build_dataset import _strict_row_passes, _train_floor_row_passes

        row = {
            "review_text": "Generic positive review about reliability in legal domain sample 11.",
            "source_text": "Generic positive review about reliability in legal domain sample 11.",
            "domain": "legal",
            "implicit": {
                "aspects": ["reliability"],
                "spans": [
                    {
                        "support_type": "gold",
                    }
                ],
                "review_reason": "low_confidence",
            },
            "explicit": {"aspects": []},
        }

        self.assertFalse(_strict_row_passes(row))
        self.assertFalse(
            _train_floor_row_passes(
                row,
                candidate_aspects_by_domain={"legal": ["reliability"]},
                accepted_support_types={"gold"},
            )
        )

    def test_family_floor_uses_non_placeholder_anchor_when_only_placeholder_candidates_exist(self) -> None:
        from build_dataset import _enforce_benchmark_family_floor

        rows_by_split = {"train": [], "val": [], "test": []}

        result = _enforce_benchmark_family_floor(
            rows_by_split,
            source_domain_family_counts={"telecom": 1, "restaurant": 0, "electronics": 0},
            fallback_rows_by_family={
                "telecom": [
                    {
                        "domain": "telecom",
                        "review_text": "Generic positive review about performance in telecom domain sample 0.",
                        "source_text": "Generic positive review about performance in telecom domain sample 0.",
                        "gold_interpretations": [
                            {
                                "aspect_label": "performance",
                                "sentiment": "positive",
                                "evidence_text": "Generic positive review about performance in telecom domain sample 0.",
                            }
                        ],
                        "split": "train",
                    }
                ]
            },
            artifact_mode="research_release",
            seed=42,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["restored_rows"], 2)
        self.assertIn("telecom", result["restored_families"])
        all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
        self.assertEqual(len(all_rows), 2)
        self.assertNotIn("Generic positive review about", all_rows[0]["review_text"])

    def test_family_floor_tops_up_existing_core_family_to_two_rows(self) -> None:
        from build_dataset import _enforce_benchmark_family_floor

        rows_by_split = {
            "train": [
                {
                    "domain": "telecom",
                    "review_text": "The connection kept dropping until I restarted the modem.",
                    "source_text": "The connection kept dropping until I restarted the modem.",
                    "gold_interpretations": [
                        {
                            "aspect_label": "connectivity",
                            "sentiment": "negative",
                            "evidence_text": "connection kept dropping",
                        }
                    ],
                    "split": "train",
                }
            ],
            "val": [],
            "test": [],
        }

        result = _enforce_benchmark_family_floor(
            rows_by_split,
            source_domain_family_counts={"telecom": 3, "restaurant": 0, "electronics": 0},
            fallback_rows_by_family={
                "telecom": [
                    {
                        "domain": "telecom",
                        "review_text": "Call quality stayed clear and the signal held steady during the whole trip.",
                        "source_text": "Call quality stayed clear and the signal held steady during the whole trip.",
                        "gold_interpretations": [
                            {
                                "aspect_label": "connectivity",
                                "sentiment": "positive",
                                "evidence_text": "signal held steady",
                            }
                        ],
                        "split": "train",
                    }
                ]
            },
            artifact_mode="research_release",
            seed=42,
        )

        all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
        telecom_rows = [row for row in all_rows if row["domain"] == "telecom"]
        self.assertTrue(result["applied"])
        self.assertEqual(len(telecom_rows), 2)
        self.assertTrue(all("Generic positive review about" not in row["review_text"] for row in telecom_rows))

    def test_validation_floor_raises_too_small_val_split(self) -> None:
        from build_dataset import _apply_benchmark_validation_floor

        rows_by_split = {
            "train": [
                {
                    "instance_id": f"train-{idx}",
                    "record_id": f"train-{idx}",
                    "review_text": f"Train review {idx}",
                    "domain": "restaurant",
                    "domain_family": "restaurant",
                    "group_id": f"train-group-{idx}",
                    "gold_interpretations": [
                        {"aspect_label": "service", "sentiment": "positive", "evidence_text": f"Train review {idx}"}
                    ],
                    "implicit_grounded_interpretations": [{"aspect_label": "service"}],
                    "explicit_grounded_interpretations": [],
                    "abstain_acceptable": False,
                    "novel_acceptable": False,
                    "novel_cluster_id": None,
                    "novel_alias": None,
                    "novel_evidence_text": None,
                    "ambiguity_score": 0.5,
                    "hardness_tier": "H1",
                    "annotation_source": "imported",
                    "split_protocol": {"random": "train", "grouped": "train", "domain_holdout": "train"},
                    "split": "train",
                }
                for idx in range(10)
            ],
            "val": [
                {
                    "instance_id": "val-1",
                    "record_id": "val-1",
                    "review_text": "Val review 1",
                    "domain": "electronics",
                    "domain_family": "electronics",
                    "group_id": "val-group-1",
                    "gold_interpretations": [
                        {"aspect_label": "power", "sentiment": "negative", "evidence_text": "Val review 1"}
                    ],
                    "implicit_grounded_interpretations": [{"aspect_label": "power"}],
                    "explicit_grounded_interpretations": [],
                    "abstain_acceptable": False,
                    "novel_acceptable": False,
                    "novel_cluster_id": None,
                    "novel_alias": None,
                    "novel_evidence_text": None,
                    "ambiguity_score": 0.5,
                    "hardness_tier": "H2",
                    "annotation_source": "imported",
                    "split_protocol": {"random": "val", "grouped": "val", "domain_holdout": "val"},
                    "split": "val",
                }
            ],
            "test": [
                {
                    "instance_id": f"test-{idx}",
                    "record_id": f"test-{idx}",
                    "review_text": f"Test review {idx}",
                    "domain": "telecom",
                    "domain_family": "telecom",
                    "group_id": f"test-group-{idx}",
                    "gold_interpretations": [
                        {"aspect_label": "connectivity", "sentiment": "positive", "evidence_text": f"Test review {idx}"}
                    ],
                    "implicit_grounded_interpretations": [{"aspect_label": "connectivity"}],
                    "explicit_grounded_interpretations": [],
                    "abstain_acceptable": False,
                    "novel_acceptable": False,
                    "novel_cluster_id": None,
                    "novel_alias": None,
                    "novel_evidence_text": None,
                    "ambiguity_score": 0.5,
                    "hardness_tier": "H3",
                    "annotation_source": "imported",
                    "split_protocol": {"random": "test", "grouped": "test", "domain_holdout": "test"},
                    "split": "test",
                }
                for idx in range(2)
            ],
        }

        result = _apply_benchmark_validation_floor(rows_by_split, seed=42, min_val_ratio=0.15)

        self.assertTrue(result["applied"])
        self.assertGreaterEqual(len(rows_by_split["val"]), 2)
        self.assertEqual(result["moved_rows"], 1)


if __name__ == "__main__":
    unittest.main()
