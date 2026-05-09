from collections import Counter
from typing import Any
from ..schemas.reports import QualityReport


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_quality_report(
    splits: dict[str, list[object]], 
    requested_rows: int = 0,
    loaded_rows: int = 0,
    processed_rows: int = 0,
    rejected_rows: int = 0,
    discarded_rows: int = 0,
    runtime_reason_counts: dict[str, int] | None = None,
    original_sample_size: int = 0,
) -> QualityReport:
    counts = {split: len(rows) for split, rows in splits.items()}
    rejected_interps = 0
    reason_counts = Counter()
    dropped_reason_counts = Counter()
    source_types = Counter()
    label_types = Counter()
    mapping_sources = Counter()
    mapping_scopes = Counter()
    mapping_layers = Counter()
    novelty = Counter()
    ambiguity_level = Counter()
    hardness = Counter()
    evidence_total = 0
    evidence_exact = 0
    full_review_evidence = 0
    matched_term_total = 0
    matched_term_hit = 0
    anchor_modifier_count = 0
    row_metadata_unknown_count = 0
    abstain_acceptable_count = 0
    generic_parent_filled = 0
    unknown_canonicals = 0
    evidence_scope_dist = Counter()
    abstain_reason_dist = Counter()
    total_gold = 0
    max_gold = 0
    for rows in splits.values():
        for row in rows:
            novelty[str(_get_val(row, "novelty_status", "known") or "known")] += 1
            ambiguity_level[str(_get_val(row, "ambiguity_level", "low") or "low")] += 1
            hardness[str(_get_val(row, "hardness_tier", "H0") or "H0")] += 1
            
            # Abstention tracking
            for reason in tuple(_get_val(row, "abstain_reason_gold", ()) or ()):
                abstain_reason_dist[str(reason)] += 1
            
            row_source_type = str(_get_val(row, "row_source_type", "unknown") or "unknown")
            row_mapping_scope = str(_get_val(row, "row_mapping_scope", "unknown") or "unknown")
            row_mapping_sources = tuple(_get_val(row, "row_mapping_sources", ()) or ())
            if (
                row_source_type == "unknown"
                or row_mapping_scope == "unknown"
                or not row_mapping_sources
            ):
                row_metadata_unknown_count += 1
            if bool(_get_val(row, "abstain_acceptable", False)):
                abstain_acceptable_count += 1
            review_text = str(_get_val(row, "review_text", "") or "")
            gold = list(_get_val(row, "gold_interpretations", []) or [])
            total_gold += len(gold)
            max_gold = max(max_gold, len(gold))
            for interp in gold:
                source_types[str(_get_val(interp, "source_type", "unknown") or "unknown")] += 1
                label_types[str(_get_val(interp, "label_type", "unknown") or "unknown")] += 1
                mapping_sources[str(_get_val(interp, "mapping_source", "none") or "none")] += 1
                if str(_get_val(interp, "mapping_source", "") or "") == "anchor_modifier":
                    anchor_modifier_count += 1
                mapping_scopes[str(_get_val(interp, "mapping_scope", "unknown") or "unknown")] += 1
                
                # Evidence Scope tracking
                scope = str(_get_val(interp, "evidence_scope", "unknown") or "unknown")
                evidence_scope_dist[scope] += 1
                
                if str(_get_val(interp, "generic_parent", "") or ""):
                    generic_parent_filled += 1
                for layer in tuple(_get_val(interp, "mapping_layers", ()) or ()):
                    mapping_layers[str(layer)] += 1
                if str(_get_val(interp, "aspect_canonical", "") or "") == "unknown":
                    unknown_canonicals += 1
                span = list(_get_val(interp, "evidence_span", []) or [])
                evidence_text = str(_get_val(interp, "evidence_text", "") or "")
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
                terms = tuple(_get_val(interp, "matched_terms", ()) or ())
                if terms:
                    matched_term_total += 1
                    ev_low = evidence_text.lower()
                    if any(str(t).lower() in ev_low for t in terms if str(t).strip()):
                        matched_term_hit += 1
                
                quality_flags = _get_val(interp, "quality_flags", []) or []
                for flag in quality_flags:
                    if flag in ("llm_drop", "repair_failed", "low_quality"):
                        rejected_interps += 1
                        reason_counts[flag] += 1
                        dropped_reason_counts[flag] += 1
                            
    total_exported = sum(counts.values())
    if runtime_reason_counts:
        for key, val in runtime_reason_counts.items():
            reason_counts[str(key)] += int(val)
            dropped_reason_counts[str(key)] += int(val)

    row_reason_counts = {}
    if runtime_reason_counts:
        row_reason_counts = {str(key): int(val) for key, val in runtime_reason_counts.items()}
    elif rejected_rows > 0:
        row_reason_counts["empty_gold_after_canonicalization"] = int(rejected_rows)

    return QualityReport(
        total_exported=total_exported, 
        export_counts=counts,
        requested_rows=requested_rows,
        loaded_rows=loaded_rows,
        processed_rows=processed_rows,
        rejected_rows=rejected_rows,
        discarded_rows=discarded_rows,
        mapping_source_distribution=dict(mapping_sources),
        mapping_scope_distribution=dict(mapping_scopes),
        mapping_layer_distribution=dict(mapping_layers),
        evidence_scope_distribution=dict(evidence_scope_dist),
        abstain_reason_distribution=dict(abstain_reason_dist),
        original_sample_size=original_sample_size,
        rejected_interpretations=rejected_interps,
        reason_counts=dict(reason_counts),
        row_rejection_reason_counts=row_reason_counts,
        dropped_interpretation_reason_counts=dict(dropped_reason_counts),
        source_type_distribution=dict(source_types),
        label_type_distribution=dict(label_types),
        novelty_distribution=dict(novelty),
        ambiguity_level_distribution=dict(ambiguity_level),
        hardness_distribution=dict(hardness),
        evidence={
            "exact_match_rate": evidence_exact / max(1, evidence_total),
            "full_review_evidence_rate": full_review_evidence / max(1, evidence_total),
            "matched_term_in_evidence_rate": matched_term_hit / max(1, matched_term_total),
            "matched_term_missing_count": max(0, matched_term_total - matched_term_hit),
        },
        canonicalization={
            "unknown_rate": unknown_canonicals / max(1, total_gold),
            "mapping_scope_unknown_count": mapping_scopes.get("unknown", 0),
            "provisional_rate": mapping_sources.get("provisional", 0) / max(1, total_gold),
            "anchor_modifier_count": anchor_modifier_count,
            "row_metadata_unknown_count": row_metadata_unknown_count,
            "generic_parent_fill_rate": generic_parent_filled / max(1, total_gold),
            "abstain_acceptable_count": abstain_acceptable_count,
            "memory_precision_audit": None,
        },
        gold_stats={
            "avg_gold_per_row": total_gold / max(1, total_exported),
            "max_gold_per_row": float(max_gold),
        },
        accounting_valid=(loaded_rows == total_exported + rejected_rows + discarded_rows),
    )
