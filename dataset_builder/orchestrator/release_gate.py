from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..reports.quality_report import build_quality_report
from ..split.leakage_checks import check_cross_split_leakage
from .exceptions import QualityGateError

def run_release_gate(output_dir: Path, cfg: Any) -> tuple[bool, dict[str, Any]]:
    """Generate reports and verify the artifact is ready for release."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = output_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                splits[split] = [json.loads(line) for line in f]
        except Exception:
            continue
            
    if not splits:
        return False, {"error": "no splits found"}
        
    # 1. Leakage Checks
    leakage = check_cross_split_leakage(splits)
    
    # 2. Quality Report
    report = build_quality_report(splits)
    
    metrics = {
        "total_rows": sum(len(r) for r in splits.values()),
        "leakage": leakage,
        "quality": report.__dict__ if hasattr(report, "__dict__") else report
    }
    
    try:
        # Use diagnostic_strict if requested in cfg, otherwise use the configured profile
        profile = "diagnostic_strict" if getattr(cfg, "strict", False) else getattr(cfg, "profile", "development")
        gate_results = assert_release_ready(splits, reports={"quality": metrics["quality"]}, leakage=leakage, profile=profile)
        metrics["gate_results"] = gate_results
        return True, metrics
    except Exception as e:
        metrics["error"] = str(e)
        return False, metrics

def assert_release_ready(
    splits: dict[str, list[Any]],
    *,
    reports: dict[str, Any],
    leakage: dict[str, int],
    profile: str = "research_default"
) -> dict[str, Any]:
    """
    Verify the artifact is ready for release based on the selected profile.
    Returns a gate_results dict.
    """
    total = sum(len(rows) for rows in splits.values())
    if total <= 0:
        raise ValueError("benchmark export is empty")
    
    # Critical Invariants (Fail regardless of profile)
    if int(leakage.get("grouped_leakage", 0)) != 0:
        gate_results = {
            "status": "FATAL",
            "profile": profile,
            "failures": ["Critical Failure: grouped split leakage detected"],
            "warnings": [],
            "metrics": {}
        }
        raise QualityGateError(gate_results, "Critical Failure: grouped split leakage detected")
    if int(leakage.get("exact_text_leakage", 0)) != 0:
        gate_results = {
            "status": "FATAL",
            "profile": profile,
            "failures": ["Critical Failure: exact text leakage detected"],
            "warnings": [],
            "metrics": {}
        }
        raise QualityGateError(gate_results, "Critical Failure: exact text leakage detected")
        
    quality = reports.get("quality", {})
    q_data = quality.__dict__ if hasattr(quality, "__dict__") else quality
    
    if not q_data.get("accounting_valid", True):
        gate_results = {
            "status": "FATAL",
            "profile": profile,
            "failures": ["Critical Failure: export accounting mismatch"],
            "warnings": [],
            "metrics": {}
        }
        raise QualityGateError(gate_results, "Critical Failure: export accounting mismatch")
    
    # Normalize evidence data
    evidence = q_data.get("evidence", {}) or {}
    match_rate = float(evidence.get("exact_match_rate", 1.0))
    if match_rate < 1.0:
        msg = f"Critical Failure: evidence exact-match rate below 100% ({match_rate})"
        gate_results = {
            "status": "FATAL",
            "profile": profile,
            "failures": [msg],
            "warnings": [],
            "metrics": {}
        }
        raise QualityGateError(gate_results, msg)

    invalid_source_types = _invalid_source_types(splits)
    if invalid_source_types:
        msg = f"Critical Failure: invalid source_type values found: {', '.join(sorted(invalid_source_types))}"
        gate_results = {
            "status": "FATAL",
            "profile": profile,
            "failures": [msg],
            "warnings": [],
            "metrics": {}
        }
        raise QualityGateError(gate_results, msg)

    # Profile-specific checks
    gate_status = "PASS"
    failures = []
    warnings = []

    # 1. Quality Issues -> FAIL (Research) or FATAL (Strict)
    unknown_count = q_data.get("canonicalization", {}).get("unknown_rate", 0.0)
    if unknown_count > 0:
        msg = f"unknown canonicals detected (rate: {unknown_count:.2%})"
        if profile == "diagnostic_strict":
            gate_results = {
                "status": "FAIL",
                "profile": profile,
                "failures": [msg],
                "warnings": [],
                "metrics": {"unknown_canonical_rate": unknown_count}
            }
            raise QualityGateError(gate_results, f"CRITICAL FAIL: {msg}")
        failures.append(msg)

    mapping_scope_unknown_count = int(q_data.get("canonicalization", {}).get("mapping_scope_unknown_count", 0))
    if mapping_scope_unknown_count > 0:
        msg = f"mapping_scope unknown detected ({mapping_scope_unknown_count})"
        failures.append(msg)
    row_metadata_unknown_count = int(q_data.get("canonicalization", {}).get("row_metadata_unknown_count", 0))
    if row_metadata_unknown_count > 0:
        msg = f"row metadata unknown detected ({row_metadata_unknown_count})"
        failures.append(msg)
    rejected_rows = int(q_data.get("rejected_rows", 0) or 0)
    reason_counts = q_data.get("reason_counts", {}) or {}
    row_rejection_reason_counts = q_data.get("row_rejection_reason_counts", {}) or {}
    dropped_interpretation_reason_counts = q_data.get("dropped_interpretation_reason_counts", {}) or {}
    has_rejection_counts = bool(reason_counts) or bool(row_rejection_reason_counts) or bool(dropped_interpretation_reason_counts)
    if rejected_rows > 0 and not has_rejection_counts:
        msg = "rejected_rows present but reason_counts is empty"
        if profile == "diagnostic_strict":
            failures.append(msg)
        else:
            warnings.append(msg)

    provisional_rate = float(q_data.get("canonicalization", {}).get("provisional_rate", 0.0))
    anchor_modifier_count = int(q_data.get("canonicalization", {}).get("anchor_modifier_count", 0))
    full_review_evidence_rate = float(evidence.get("full_review_evidence_rate", 0.0))
    matched_term_in_evidence_rate = float(evidence.get("matched_term_in_evidence_rate", 1.0))

    if profile == "diagnostic_strict":
        if provisional_rate > 0.25:
            failures.append(f"provisional rate too high ({provisional_rate:.2%})")
        if anchor_modifier_count == 0:
            failures.append("anchor_modifier_count is zero")
        if full_review_evidence_rate > 0.20:
            failures.append(f"full_review_evidence_rate too high ({full_review_evidence_rate:.2%})")
        if matched_term_in_evidence_rate < 0.95:
            warnings.append(f"matched_term_in_evidence_rate warning ({matched_term_in_evidence_rate:.2%})")
    else:
        if provisional_rate > 0.70:
            failures.append(f"provisional rate too high ({provisional_rate:.2%})")
        elif provisional_rate > 0.50:
            warnings.append(f"provisional rate warning ({provisional_rate:.2%})")
        if full_review_evidence_rate > 0.30:
            failures.append(f"full_review_evidence_rate too high ({full_review_evidence_rate:.2%})")
        elif full_review_evidence_rate > 0.15:
            warnings.append(f"full_review_evidence_rate warning ({full_review_evidence_rate:.2%})")
        if anchor_modifier_count == 0:
            warnings.append("anchor_modifier_count is zero")
        if matched_term_in_evidence_rate < 0.85:
            warnings.append(f"matched_term_in_evidence_rate warning ({matched_term_in_evidence_rate:.2%})")
        
    # 2. Novelty Overfiring -> FAIL
    novelty_dist = q_data.get("novelty_distribution", {})
    novel_rows = novelty_dist.get("novel", 0)
    novelty_rate = novel_rows / total
    if profile == "diagnostic_strict" and novelty_rate > 0.5:
        msg = f"novelty rate too high ({novelty_rate:.2%}) for diagnostic run"
        gate_results = {
            "status": "FAIL",
            "profile": profile,
            "failures": [msg],
            "warnings": [],
            "metrics": {"novelty_rate": novelty_rate}
        }
        raise QualityGateError(gate_results, f"CRITICAL FAIL: {msg}")
    elif novelty_rate > 0.8:
        failures.append(f"extreme novelty rate detected: {novelty_rate:.2%}")

    # 3. Mapping Provenance -> FATAL in Strict
    mapping_dist = q_data.get("mapping_source_distribution", {})
    unknown_provenance = mapping_dist.get("unknown", 0) + mapping_dist.get("none", 0)
    provenance_unknown_rate = unknown_provenance / max(1, sum(mapping_dist.values()))
    if profile == "diagnostic_strict" and provenance_unknown_rate > 0:
        msg = f"{unknown_provenance} interpretations have unknown mapping provenance"
        gate_results = {
            "status": "FAIL",
            "profile": profile,
            "failures": [msg],
            "warnings": [],
            "metrics": {"provenance_unknown_rate": provenance_unknown_rate}
        }
        raise QualityGateError(gate_results, f"CRITICAL FAIL: {msg}")
    elif provenance_unknown_rate > 0.1:
        failures.append(f"high unknown mapping provenance: {provenance_unknown_rate:.2%}")

    # 4. Source Distribution -> FAIL
    source_dist = q_data.get("source_type_distribution", {})
    require_learned = bool(reports.get("require_learned", False))
    if require_learned and int(source_dist.get("implicit_learned", 0)) <= 0:
        failures.append("zero implicit_learned interpretations in learned run")

    if failures:
        gate_status = "FAIL"
        if profile == "diagnostic_strict":
             gate_results = {
                 "status": gate_status,
                 "profile": profile,
                 "failures": failures,
                 "warnings": warnings,
                "metrics": {
                    "novelty_rate": novelty_rate,
                    "provenance_unknown_rate": provenance_unknown_rate,
                    "unknown_canonical_rate": unknown_count,
                    "mapping_scope_unknown_count": mapping_scope_unknown_count,
                    "row_metadata_unknown_count": row_metadata_unknown_count,
                    "provisional_rate": provisional_rate,
                    "anchor_modifier_count": anchor_modifier_count,
                    "full_review_evidence_rate": full_review_evidence_rate,
                    "matched_term_in_evidence_rate": matched_term_in_evidence_rate,
                }
             }
             raise QualityGateError(gate_results, f"Gate Failures ({profile}): " + "; ".join(failures))

    return {
        "status": gate_status,
        "profile": profile,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "novelty_rate": novelty_rate,
            "provenance_unknown_rate": provenance_unknown_rate,
            "unknown_canonical_rate": unknown_count,
            "mapping_scope_unknown_count": mapping_scope_unknown_count,
            "row_metadata_unknown_count": row_metadata_unknown_count,
            "provisional_rate": provisional_rate,
            "anchor_modifier_count": anchor_modifier_count,
            "full_review_evidence_rate": full_review_evidence_rate,
            "matched_term_in_evidence_rate": matched_term_in_evidence_rate,
        }
    }

def _invalid_source_types(splits: dict[str, list[Any]]) -> set[str]:
    valid = {"explicit", "implicit_learned", "implicit_json", "implicit_llm", "merged"}
    invalid: set[str] = set()
    for rows in splits.values():
        for row in rows:
            interps = _get_value(row, "gold_interpretations", []) or []
            for interp in interps:
                source_type = str(_get_value(interp, "source_type", "unknown") or "unknown")
                if source_type not in valid:
                    invalid.add(source_type)
    return invalid

def _get_value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)
