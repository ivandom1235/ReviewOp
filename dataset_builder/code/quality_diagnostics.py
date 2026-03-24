from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from evidence_extract import extract_evidence
from mappings import EXPORT_CANONICAL_ASPECTS
from utils import normalize_text, stable_hash


GENERIC_ASPECT_BUCKETS = {
    "miscellaneous",
    "generic",
    "quality",
    "experience",
}


def _label_key(label: Dict[str, Any]) -> str:
    aspect = str(label.get("aspect", "unknown")).strip() or "unknown"
    sentiment = str(label.get("sentiment", "neutral")).strip().lower() or "neutral"
    return f"{aspect}__{sentiment}"


def _label_issue_reasons(label: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    aspect = str(label.get("aspect", "")).strip()
    evidence_sentence = str(label.get("evidence_sentence", "")).strip()
    review_text = str(row.get("review_text", "")).strip()
    confidence = float(label.get("confidence", 0.0))
    metadata = dict(label.get("metadata", {}))
    evidence_aspect = str(metadata.get("raw_aspect") or metadata.get("aspect_surface") or aspect).strip()

    if metadata.get("augmentation_sentence_evidence"):
        evidence_info = {
            "evidence_text": evidence_sentence,
            "evidence_quality": float(metadata.get("evidence_quality", 0.92) or 0.92),
            "is_sentence_fallback": False,
        }
    else:
        evidence_info = extract_evidence(
            text=review_text,
            evidence_sentence=evidence_sentence,
            aspect_raw=evidence_aspect,
        )

    if confidence < 0.55:
        reasons.append("low_confidence")
    if evidence_info["is_sentence_fallback"]:
        reasons.append("sentence_level_evidence")
    if evidence_sentence and review_text and normalize_text(evidence_sentence) == normalize_text(review_text):
        reasons.append("whole_review_evidence")
    if evidence_info["evidence_quality"] < 0.6:
        reasons.append("weak_evidence_quality")
    if aspect in GENERIC_ASPECT_BUCKETS or aspect.startswith("other_"):
        reasons.append("generic_or_other_aspect")
    if aspect and aspect not in EXPORT_CANONICAL_ASPECTS:
        reasons.append("noncanonical_export_aspect")
    if label.get("type") == "implicit" and confidence < 0.7:
        reasons.append("weak_implicit_label")
    if metadata.get("mapping_mode") in {"open_aspect", "open_aspect_compacted"}:
        reasons.append("open_aspect_mapping")
    return reasons


def build_data_quality_report(
    review_rows: List[Dict[str, Any]],
    episodic_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    label_type_counts = Counter()
    joint_counts = Counter()
    aspect_counts = Counter()
    surface_counts = Counter()
    evidence_stats = Counter()
    multi_label_hist = Counter()
    domain_counts = Counter()
    mapping_mode_counts = Counter()
    aspect_family_counts = Counter()
    implicit_by_domain = Counter()
    senticnet_hits = 0
    noncanonical_aspects = Counter()
    confidence_values: List[float] = []

    for row in review_rows:
        labels = list(row.get("labels", []))
        multi_label_hist[len(labels)] += 1
        domain_counts[str(row.get("domain", "unknown")).strip().lower() or "unknown"] += 1
        for label in labels:
            label_type_counts[str(label.get("type", "unknown")).strip().lower() or "unknown"] += 1
            aspect = str(label.get("aspect", "unknown")).strip() or "unknown"
            aspect_counts[aspect] += 1
            if aspect not in EXPORT_CANONICAL_ASPECTS:
                noncanonical_aspects[aspect] += 1
            joint_counts[_label_key(label)] += 1
            confidence_values.append(float(label.get("confidence", 0.0)))
            metadata = dict(label.get("metadata", {}))
            surface = str(metadata.get("aspect_surface", "")).strip()
            if surface:
                surface_counts[surface] += 1
            mapping_mode = str(metadata.get("mapping_mode", "")).strip()
            if mapping_mode:
                mapping_mode_counts[mapping_mode] += 1
            aspect_family = str(metadata.get("aspect_family", "")).strip()
            if aspect_family:
                aspect_family_counts[aspect_family] += 1
            if str(label.get("type", "")).strip().lower() == "implicit":
                implicit_by_domain[str(row.get("domain", "unknown")).strip().lower() or "unknown"] += 1
            if metadata.get("senticnet_concept"):
                senticnet_hits += 1
            evidence_sentence = str(label.get("evidence_sentence", "")).strip()
            review_text = str(row.get("review_text", "")).strip()
            evidence_aspect = str(metadata.get("raw_aspect") or metadata.get("aspect_surface") or aspect).strip()
            if metadata.get("augmentation_sentence_evidence"):
                evidence_info = {
                    "evidence_text": evidence_sentence,
                    "evidence_quality": float(metadata.get("evidence_quality", 0.92) or 0.92),
                    "is_sentence_fallback": False,
                }
            else:
                evidence_info = extract_evidence(
                    text=review_text,
                    evidence_sentence=evidence_sentence,
                    aspect_raw=evidence_aspect,
                )
            if evidence_info["is_sentence_fallback"]:
                evidence_stats["sentence_fallback"] += 1
            if evidence_info["evidence_quality"] < 0.6:
                evidence_stats["low_quality"] += 1
            if review_text and normalize_text(review_text) == normalize_text(evidence_sentence):
                evidence_stats["whole_review"] += 1
            if len(str(evidence_info["evidence_text"]).split()) <= 3:
                evidence_stats["very_short"] += 1

    n_labels = max(1, sum(label_type_counts.values()))
    n_reviews = max(1, len(review_rows))
    return {
        "review_rows": len(review_rows),
        "episodic_rows": len(episodic_rows),
        "domains": dict(domain_counts),
        "label_type_counts": dict(label_type_counts),
        "implicit_ratio": round(label_type_counts.get("implicit", 0) / n_labels, 4),
        "avg_labels_per_review": round(sum(k * v for k, v in multi_label_hist.items()) / n_reviews, 4),
        "reviews_with_2plus_labels": sum(v for k, v in multi_label_hist.items() if k >= 2),
        "multi_label_histogram": dict(sorted(multi_label_hist.items())),
        "top_aspects": aspect_counts.most_common(20),
        "top_surface_aspects": surface_counts.most_common(20),
        "implicit_by_domain": dict(implicit_by_domain),
        "joint_label_inventory_size": len(joint_counts),
        "joint_labels_below_5": sum(1 for c in joint_counts.values() if c < 5),
        "joint_labels_below_10": sum(1 for c in joint_counts.values() if c < 10),
        "canonical_label_inventory_size": len([aspect for aspect in aspect_counts if aspect in EXPORT_CANONICAL_ASPECTS]),
        "surface_label_leak_count": sum(noncanonical_aspects.values()),
        "surface_label_leaks": noncanonical_aspects.most_common(20),
        "other_bucket_rate": round(sum(count for aspect, count in aspect_counts.items() if str(aspect).startswith("other_")) / n_labels, 4),
        "mapping_modes": dict(mapping_mode_counts),
        "aspect_families": dict(aspect_family_counts),
        "broad_label_rate": round(sum(count for aspect, count in aspect_counts.items() if aspect in {"product_quality", "performance", "support_quality", "value", "usability"}) / n_labels, 4),
        "senticnet_hit_rate": round(senticnet_hits / n_labels, 4),
        "multi_aspect_rate": round(sum(v for k, v in multi_label_hist.items() if k >= 2) / n_reviews, 4),
        "evidence": {
            "whole_review_rate": round(evidence_stats.get("whole_review", 0) / n_labels, 4),
            "sentence_fallback_rate": round(evidence_stats.get("sentence_fallback", 0) / n_labels, 4),
            "low_quality_rate": round(evidence_stats.get("low_quality", 0) / n_labels, 4),
            "very_short_rate": round(evidence_stats.get("very_short", 0) / n_labels, 4),
        },
        "confidence": {
            "mean": round(sum(confidence_values) / max(1, len(confidence_values)), 4),
            "min": round(min(confidence_values), 4) if confidence_values else 0.0,
            "max": round(max(confidence_values), 4) if confidence_values else 0.0,
        },
    }


def build_label_issue_candidates(review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issue_rows: List[Dict[str, Any]] = []
    for row in review_rows:
        labels = list(row.get("labels", []))
        aspect_to_sentiments = defaultdict(set)
        for label in labels:
            aspect_to_sentiments[str(label.get("aspect", ""))].add(str(label.get("sentiment", "neutral")).lower())

        contradictory_aspects = sorted(aspect for aspect, sentiments in aspect_to_sentiments.items() if len(sentiments) > 1)
        for index, label in enumerate(labels):
            reasons = _label_issue_reasons(label, row)
            if contradictory_aspects and str(label.get("aspect", "")) in contradictory_aspects:
                reasons.append("contradictory_aspect_sentiment")
            if not reasons:
                continue
            issue_rows.append(
                {
                    "issue_id": stable_hash(str(row.get("id", "")), str(index), str(label.get("aspect", "")), str(label.get("sentiment", ""))),
                    "row_id": row.get("id"),
                    "domain": row.get("domain", "unknown"),
                    "aspect": label.get("aspect"),
                    "sentiment": label.get("sentiment"),
                    "type": label.get("type", "unknown"),
                    "confidence": float(label.get("confidence", 0.0)),
                    "reasons": sorted(set(reasons)),
                    "evidence_sentence": label.get("evidence_sentence", ""),
                    "review_text": row.get("review_text", ""),
                }
            )
    issue_rows.sort(key=lambda item: (-len(item["reasons"]), item["confidence"], str(item["row_id"])))
    return issue_rows


def build_episode_readiness_report(
    episodic_rows: Iterable[Dict[str, Any]],
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Dict[str, Any]:
    needed = k_shot + q_query
    rows = list(episodic_rows)
    by_split_and_joint: Dict[str, Counter] = defaultdict(Counter)
    split_parents: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    by_split_and_grouped: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split", "train")).lower()
        aspect = str(row.get("aspect", "unknown")).strip() or "unknown"
        sentiment = str(row.get("sentiment", "neutral")).strip().lower() or "neutral"
        key = f"{aspect}__{sentiment}"
        by_split_and_joint[split][key] += 1
        split_parents[split][key].add(str(row.get("parent_review_id", row.get("example_id", ""))))
        metadata = dict(row.get("metadata", {}))
        grouped_aspect = str(metadata.get("aspect_family", "")).strip() or aspect
        grouped_key = f"{grouped_aspect}__{sentiment}"
        by_split_and_grouped[split][grouped_key] += 1

    report: Dict[str, Any] = {
        "config": {"n_way": n_way, "k_shot": k_shot, "q_query": q_query, "needed_unique_reviews_per_joint_label": needed},
        "splits": {},
    }
    for split in sorted(by_split_and_joint):
        counts = by_split_and_joint[split]
        safe_labels = sorted(
            label for label, count in counts.items()
            if len(split_parents[split][label]) >= needed
        )
        report["splits"][split] = {
            "joint_label_count": len(counts),
            "safe_joint_label_count": len(safe_labels),
            "can_support_config": len(safe_labels) >= n_way,
            "grouped_joint_label_count": len(by_split_and_grouped[split]),
            "safe_joint_labels": safe_labels[:50],
            "unsafe_joint_labels": [
                {
                    "joint_label": label,
                    "example_count": count,
                    "unique_parent_reviews": len(split_parents[split][label]),
                }
                for label, count in sorted(counts.items())
                if len(split_parents[split][label]) < needed
            ][:50],
        }
    return report
