from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ..reports.quality_report import build_quality_report
from ..split.leakage_checks import check_cross_split_leakage

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
        # Use diagnostic_strict if requested in cfg
        profile = "diagnostic_strict" if getattr(cfg, "strict", False) else "research_default"
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
        raise ValueError("Critical Failure: grouped split leakage detected")
    if int(leakage.get("exact_text_leakage", 0)) != 0:
        raise ValueError("Critical Failure: exact text leakage detected")
        
    quality = reports.get("quality", {})
    q_data = quality.__dict__ if hasattr(quality, "__dict__") else quality
    
    if not q_data.get("accounting_valid", True):
        raise ValueError("Critical Failure: export accounting mismatch")
    
    # Normalize evidence data
    evidence = q_data.get("evidence", {}) or {}
    match_rate = float(evidence.get("exact_match_rate", 1.0))
    if match_rate < 1.0:
        raise ValueError(f"Critical Failure: evidence exact-match rate below 100% ({match_rate})")

    invalid_source_types = _invalid_source_types(splits)
    if invalid_source_types:
        raise ValueError(f"Critical Failure: invalid source_type values found: {', '.join(sorted(invalid_source_types))}")

    # Profile-specific checks
    gate_status = "PASS"
    failures = []
    warnings = []

    # 1. Quality Issues -> FAIL (Research) or FATAL (Strict)
    unknown_count = q_data.get("canonicalization", {}).get("unknown_rate", 0.0)
    if unknown_count > 0:
        msg = f"unknown canonicals detected (rate: {unknown_count:.2%})"
        if profile == "diagnostic_strict":
            raise ValueError(f"CRITICAL FAIL: {msg}")
        failures.append(msg)
        
    # 2. Novelty Overfiring -> FAIL
    novelty_dist = q_data.get("novelty_distribution", {})
    novel_rows = novelty_dist.get("novel", 0)
    novelty_rate = novel_rows / total
    if profile == "diagnostic_strict" and novelty_rate > 0.5:
        raise ValueError(f"CRITICAL FAIL: novelty rate too high ({novelty_rate:.2%}) for diagnostic run")
    elif novelty_rate > 0.8:
        failures.append(f"extreme novelty rate detected: {novelty_rate:.2%}")

    # 3. Mapping Provenance -> FATAL in Strict
    mapping_dist = q_data.get("mapping_source_distribution", {})
    unknown_provenance = mapping_dist.get("unknown", 0) + mapping_dist.get("none", 0)
    provenance_unknown_rate = unknown_provenance / max(1, sum(mapping_dist.values()))
    if profile == "diagnostic_strict" and provenance_unknown_rate > 0:
        raise ValueError(f"CRITICAL FAIL: {unknown_provenance} interpretations have unknown mapping provenance")
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
             raise ValueError(f"Gate Failures ({profile}): " + "; ".join(failures))

    return {
        "status": gate_status,
        "profile": profile,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "novelty_rate": novelty_rate,
            "provenance_unknown_rate": provenance_unknown_rate,
            "unknown_canonical_rate": unknown_count
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
