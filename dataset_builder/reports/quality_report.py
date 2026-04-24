from __future__ import annotations

from collections import Counter
from ..schemas.reports import QualityReport


def build_quality_report(
    splits: dict[str, list[object]], 
    requested_rows: int = 0,
    loaded_rows: int = 0,
    processed_rows: int = 0,
    rejected_rows: int = 0,
    discarded_rows: int = 0
) -> QualityReport:
    counts = {split: len(rows) for split, rows in splits.items()}
    rejected_interps = 0
    reason_counts = Counter()
    source_types = Counter()
    label_types = Counter()
    mapping_sources = Counter()
    novelty = Counter()
    hardness = Counter()
    evidence_total = 0
    evidence_exact = 0
    full_review_evidence = 0
    unknown_canonicals = 0
    total_gold = 0
    max_gold = 0
    for rows in splits.values():
        for row in rows:
            novelty[str(getattr(row, "novelty_status", "known") or "known")] += 1
            hardness[str(getattr(row, "hardness_tier", "H0") or "H0")] += 1
            review_text = str(getattr(row, "review_text", "") or "")
            gold = list(getattr(row, "gold_interpretations", []) or []) if hasattr(row, "gold_interpretations") else []
            total_gold += len(gold)
            max_gold = max(max_gold, len(gold))
            if not hasattr(row, "gold_interpretations"):
                continue
            for interp in getattr(row, "gold_interpretations", []):
                source_types[str(getattr(interp, "source_type", "unknown") or "unknown")] += 1
                label_types[str(getattr(interp, "label_type", "unknown") or "unknown")] += 1
                mapping_sources[str(getattr(interp, "mapping_source", "none") or "none")] += 1
                if str(getattr(interp, "aspect_canonical", "") or "") == "unknown":
                    unknown_canonicals += 1
                span = list(getattr(interp, "evidence_span", []) or [])
                evidence_text = str(getattr(interp, "evidence_text", "") or "")
                if len(span) == 2:
                    evidence_total += 1
                    try:
                        start, end = int(span[0]), int(span[1])
                        if review_text[start:end] == evidence_text:
                            evidence_exact += 1
                        if start == 0 and end == len(review_text):
                            full_review_evidence += 1
                    except (TypeError, ValueError):
                        pass
                if interp and hasattr(interp, "quality_flags"):
                    for flag in interp.quality_flags:
                        if flag in ("llm_drop", "repair_failed", "low_quality"):
                            rejected_interps += 1
                            reason_counts[flag] += 1
                            
    total_exported = sum(counts.values())
    return QualityReport(
        total_exported=total_exported, 
        export_counts=counts,
        requested_rows=requested_rows,
        loaded_rows=loaded_rows,
        processed_rows=processed_rows,
        rejected_rows=rejected_rows,
        discarded_rows=discarded_rows,
        mapping_source_distribution=dict(mapping_sources),
        rejected_interpretations=rejected_interps,
        reason_counts=dict(reason_counts),
        source_type_distribution=dict(source_types),
        label_type_distribution=dict(label_types),
        novelty_distribution=dict(novelty),
        hardness_distribution=dict(hardness),
        evidence={
            "exact_match_rate": evidence_exact / max(1, evidence_total),
            "full_review_evidence_rate": full_review_evidence / max(1, evidence_total),
        },
        canonicalization={
            "unknown_rate": unknown_canonicals / max(1, total_gold),
        },
        gold_stats={
            "avg_gold_per_row": total_gold / max(1, total_exported),
            "max_gold_per_row": float(max_gold),
        },
        accounting_valid=(loaded_rows == total_exported + rejected_rows + discarded_rows),
    )
