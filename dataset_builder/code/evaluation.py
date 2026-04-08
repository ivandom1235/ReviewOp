from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def aspect_metrics(pred_rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    rows = list(pred_rows)
    aspect_counts: Counter[str] = Counter()
    sentiments: Counter[str] = Counter()
    for row in rows:
        implicit = row.get("implicit", {})
        for aspect in implicit.get("aspects", []):
            aspect_counts[aspect] += 1
            sentiments[str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral")))] += 1
    return {
        "num_rows": len(rows),
        "aspect_counts": dict(aspect_counts),
        "sentiment_counts": dict(sentiments),
    }


def _span_signature(span: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(span.get("latent_aspect") or span.get("aspect") or ""),
        str(span.get("surface_aspect") or span.get("matched_surface") or ""),
        int(span.get("start_char", -1) or -1),
        int(span.get("end_char", -1) or -1),
        str(span.get("sentiment") or "neutral"),
    )


def span_f1(pred_rows: Iterable[dict[str, Any]], gold_rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    pred = list(pred_rows)
    gold = list(gold_rows)
    pred_spans = set()
    gold_spans = set()
    for row in pred:
        for span in row.get("implicit", {}).get("spans", []):
            pred_spans.add((row.get("id"), *_span_signature(span)))
    for row in gold:
        for span in row.get("implicit", {}).get("spans", []):
            gold_spans.add((row.get("id"), *_span_signature(span)))
    true_positive = len(pred_spans & gold_spans)
    precision = true_positive / len(pred_spans) if pred_spans else 0.0
    recall = true_positive / len(gold_spans) if gold_spans else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive,
        "predicted": len(pred_spans),
        "gold": len(gold_spans),
    }


def benchmark_scorecard(
    pred_rows: Iterable[dict[str, Any]],
    gold_rows: Iterable[dict[str, Any]],
    *,
    benchmark_family: str,
    model_family: str,
) -> Dict[str, Any]:
    pred = list(pred_rows)
    gold = list(gold_rows)
    scorecard = {
        "benchmark_family": benchmark_family,
        "model_family": model_family,
        "span_f1": span_f1(pred, gold),
    }
    scorecard["aspect_metrics"] = aspect_metrics(pred)
    scorecard["language_breakdown"] = {}
    languages = sorted({str(row.get("language", "unknown")) for row in pred})
    for language in languages:
        pred_subset = [row for row in pred if str(row.get("language", "unknown")) == language]
        gold_subset = [row for row in gold if str(row.get("language", "unknown")) == language]
        if pred_subset or gold_subset:
            scorecard["language_breakdown"][language] = {
                "span_f1": span_f1(pred_subset, gold_subset),
                "aspect_metrics": aspect_metrics(pred_subset),
            }
    return scorecard


def _norm_aspect(value: Any) -> str:
    return str(value or "").strip().lower()


def _label_signature(label: dict[str, Any]) -> tuple[str, str]:
    return (_norm_aspect(label.get("aspect")), str(label.get("sentiment") or "neutral").lower())


def gold_eval(rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    records = list(rows)
    eligible = [row for row in records if isinstance(row.get("gold_labels"), list) and row.get("gold_labels")]
    if not eligible:
        return {
            "has_gold_labels": False,
            "num_rows_with_gold": 0,
            "aspect_f1": 0.0,
            "sentiment_f1": 0.0,
            "span_overlap_f1": 0.0,
            "by_domain": {},
        }

    def compute(subset: list[dict[str, Any]]) -> dict[str, Any]:
        gold_aspects: set[tuple[Any, str]] = set()
        pred_aspects: set[tuple[Any, str]] = set()
        gold_sentiments: set[tuple[Any, str, str]] = set()
        pred_sentiments: set[tuple[Any, str, str]] = set()
        gold_spans: set[tuple[Any, str, int, int]] = set()
        pred_spans: set[tuple[Any, str, int, int]] = set()
        for row in subset:
            row_id = row.get("id")
            for label in row.get("gold_labels", []):
                if not isinstance(label, dict):
                    continue
                aspect, sentiment = _label_signature(label)
                if aspect:
                    gold_aspects.add((row_id, aspect))
                    gold_sentiments.add((row_id, aspect, sentiment))
                    start = int(label.get("start", -1) or -1)
                    end = int(label.get("end", -1) or -1)
                    if start >= 0 and end >= 0:
                        gold_spans.add((row_id, aspect, start, end))
            implicit = row.get("implicit", {})
            for aspect in implicit.get("aspects", []):
                aspect_key = _norm_aspect(aspect)
                if aspect_key and aspect_key != "general":
                    pred_aspects.add((row_id, aspect_key))
                    sentiment = str(implicit.get("aspect_sentiments", {}).get(aspect, implicit.get("dominant_sentiment", "neutral"))).lower()
                    pred_sentiments.add((row_id, aspect_key, sentiment))
            for span in implicit.get("spans", []):
                if not isinstance(span, dict):
                    continue
                aspect_key = _norm_aspect(span.get("latent_aspect") or span.get("aspect"))
                start = int(span.get("start_char", -1) or -1)
                end = int(span.get("end_char", -1) or -1)
                if aspect_key and aspect_key != "general" and start >= 0 and end >= 0:
                    pred_spans.add((row_id, aspect_key, start, end))

        def f1(pred: set, gold: set) -> float:
            tp = len(pred & gold)
            precision = tp / len(pred) if pred else 0.0
            recall = tp / len(gold) if gold else 0.0
            return round((2 * precision * recall / (precision + recall)) if precision + recall else 0.0, 4)

        return {
            "num_rows": len(subset),
            "aspect_f1": f1(pred_aspects, gold_aspects),
            "sentiment_f1": f1(pred_sentiments, gold_sentiments),
            "span_overlap_f1": f1(pred_spans, gold_spans),
        }

    overall = compute(eligible)
    by_domain: dict[str, Any] = {}
    for domain in sorted({str(row.get("domain", "unknown")) for row in eligible}):
        domain_rows = [row for row in eligible if str(row.get("domain", "unknown")) == domain]
        by_domain[domain] = compute(domain_rows)
    return {
        "has_gold_labels": True,
        "num_rows_with_gold": len(eligible),
        "aspect_f1": overall["aspect_f1"],
        "sentiment_f1": overall["sentiment_f1"],
        "span_overlap_f1": overall["span_overlap_f1"],
        "by_domain": by_domain,
    }


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def _interpretation_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _norm_aspect(item.get("aspect_label") or item.get("aspect")),
        str(item.get("sentiment") or "neutral").strip().lower(),
        _norm_text(item.get("evidence_text") or item.get("evidence")),
    )


def benchmark_gold_eval(rows: Iterable[dict[str, Any]]) -> Dict[str, Any]:
    records = list(rows)
    eligible = [row for row in records if isinstance(row.get("gold_interpretations"), list) and row.get("gold_interpretations")]
    if not eligible:
        return {
            "has_gold_interpretations": False,
            "num_rows_with_gold_interpretations": 0,
            "total_interpretations": 0,
            "average_gold_interpretations": 0.0,
            "multi_gold_label_rate": 0.0,
            "grounded_evidence_rate": 0.0,
            "duplicate_interpretation_rate": 0.0,
            "by_domain": {},
        }

    total_interpretations = 0
    grounded_interpretations = 0
    duplicate_interpretations = 0
    seen: set[tuple[str, str, str, str]] = set()
    by_domain: dict[str, dict[str, int]] = {}
    ambiguity_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    abstain_acceptable_rows = 0
    novel_acceptable_rows = 0
    novel_cluster_counts: Counter[str] = Counter()
    explicit_grounded_total = 0
    implicit_grounded_total = 0
    fallback_only_implicit = 0
    ontology_compatible_total = 0

    for row in eligible:
        domain = str(row.get("domain", "unknown"))
        domain_stats = by_domain.setdefault(domain, {"rows": 0, "interpretations": 0, "grounded": 0})
        domain_stats["rows"] += 1
        if bool(row.get("abstain_acceptable", False)):
            abstain_acceptable_rows += 1
        if bool(row.get("novel_acceptable", False)):
            novel_acceptable_rows += 1
            cluster_id = str(row.get("novel_cluster_id") or "").strip()
            if cluster_id:
                novel_cluster_counts[cluster_id] += 1
        review_text = _norm_text(row.get("review_text") or row.get("source_text"))
        explicit_grounded_total += len(list(row.get("explicit_grounded_interpretations") or []))
        implicit_grounded_total += len(list(row.get("implicit_grounded_interpretations") or []))
        for item in row.get("gold_interpretations", []):
            if not isinstance(item, dict):
                continue
            total_interpretations += 1
            domain_stats["interpretations"] += 1
            source_counts[str(item.get("source") or item.get("annotation_source") or item.get("label_source") or "unknown").strip() or "unknown"] += 1
            signature = (domain, *_interpretation_signature(item))
            if signature in seen:
                duplicate_interpretations += 1
            else:
                seen.add(signature)
            ambiguity_type = str(item.get("ambiguity_type") or "none").strip().lower() or "none"
            ambiguity_type_counts[ambiguity_type] += 1
            evidence = _norm_text(item.get("evidence_text") or item.get("evidence"))
            if evidence and evidence in review_text:
                grounded_interpretations += 1
                domain_stats["grounded"] += 1
            if bool(item.get("fallback_used", False)):
                fallback_only_implicit += 1
            canonical = str(item.get("domain_canonical_aspect") or "").strip()
            if canonical:
                ontology_compatible_total += 1

    return {
        "has_gold_interpretations": True,
        "num_rows_with_gold_interpretations": len(eligible),
        "total_interpretations": total_interpretations,
        "average_gold_interpretations": round(total_interpretations / max(1, len(eligible)), 4),
        "multi_gold_label_rate": round(sum(1 for row in eligible if len(row.get("gold_interpretations", [])) > 1) / max(1, len(eligible)), 4),
        "grounded_evidence_rate": round(grounded_interpretations / max(1, total_interpretations), 4),
        "duplicate_interpretation_rate": round(duplicate_interpretations / max(1, total_interpretations), 4),
        "abstain_acceptable_rate": round(abstain_acceptable_rows / max(1, len(eligible)), 4),
        "novel_acceptable_rate": round(novel_acceptable_rows / max(1, len(eligible)), 4),
        "novel_cluster_count": int(len(novel_cluster_counts)),
        "novel_cluster_frequency": dict(novel_cluster_counts.most_common()),
        "implicit_purity_rate": round(implicit_grounded_total / max(1, implicit_grounded_total + explicit_grounded_total), 4),
        "fallback_only_implicit_rate": round(fallback_only_implicit / max(1, implicit_grounded_total), 4),
        "ontology_compatibility_rate": round(ontology_compatible_total / max(1, total_interpretations), 4),
        "interpretation_source_distribution": dict(source_counts.most_common()),
        "ambiguity_type_distribution": dict(ambiguity_type_counts),
        "by_domain": by_domain,
    }


def benchmark_structural_audits(rows_by_split: Dict[str, list[dict[str, Any]]]) -> Dict[str, Any]:
    split_names = ("train", "val", "test")
    all_rows = [row for split in split_names for row in rows_by_split.get(split, [])]
    total_rows = max(1, len(all_rows))
    invalid_spans = 0
    group_ids = [str(row.get("group_id") or "unknown") for row in all_rows]
    explicit_count = 0
    implicit_count = 0
    split_label_sets: Dict[str, set[str]] = {split: set() for split in split_names}
    split_h2h3: Dict[str, int] = {split: 0 for split in split_names}
    split_abstain: Dict[str, int] = {split: 0 for split in split_names}
    split_novel: Dict[str, int] = {split: 0 for split in split_names}

    for split in split_names:
        for row in rows_by_split.get(split, []):
            if bool(row.get("abstain_acceptable", False)):
                split_abstain[split] += 1
            if bool(row.get("novel_acceptable", False)):
                split_novel[split] += 1
            hardness = str(row.get("hardness_tier") or "H0").strip().upper()
            if hardness in {"H2", "H3"}:
                split_h2h3[split] += 1
            for item in list(row.get("gold_interpretations") or []):
                if not isinstance(item, dict):
                    continue
                aspect = str(item.get("aspect_label") or item.get("aspect") or "").strip().lower()
                if aspect:
                    split_label_sets[split].add(aspect)
                label_source = str(item.get("label_type") or item.get("source") or "").strip().lower()
                if label_source == "explicit":
                    explicit_count += 1
                else:
                    implicit_count += 1
                span = item.get("evidence_span")
                if not (isinstance(span, list) and len(span) == 2):
                    invalid_spans += 1
                    continue
                try:
                    start = int(span[0] if span[0] is not None else -1)
                    end = int(span[1] if span[1] is not None else -1)
                except (TypeError, ValueError):
                    invalid_spans += 1
                    continue
                review = str(row.get("review_text") or "")
                if start < 0 or end < start or end > len(review):
                    invalid_spans += 1

    union_labels = set().union(*split_label_sets.values()) if split_label_sets else set()
    protocol_label_parity = {
        split: round(len(split_label_sets[split]) / max(1, len(union_labels)), 4)
        for split in split_names
    }
    return {
        "invalid_span_rate": round(invalid_spans / total_rows, 4),
        "group_id_uniqueness_ratio": round(len(set(group_ids)) / total_rows, 4),
        "protocol_label_coverage_parity": protocol_label_parity,
        "abstain_positive_by_split": split_abstain,
        "novel_positive_by_split": split_novel,
        "h2_h3_by_split": split_h2h3,
        "source_mix": {
            "explicit": int(explicit_count),
            "implicit": int(implicit_count),
        },
    }
