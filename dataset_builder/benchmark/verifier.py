from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .verification_policy import PROFILE_DEFAULTS, profile_thresholds
logger = logging.getLogger(__name__)

def verify_artifact_dir(
    output_dir: Path,
    *,
    profile: str = "development",
    expected_rows: int | None = None,
    require_organic_memory: bool = False,
    require_rejected_row_audit: bool = False,
    require_counterfactual_source_breakdown: bool = False,
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics_summary.json"
    quality_path = output_dir / "quality_report.json"
    cf_quality_path = output_dir / "counterfactual" / "counterfactual_quality_report.json"
    rejected_rows_path = output_dir / "rejected_rows.jsonl"

    def load_json(p):
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse JSON file %s: %s", p, exc)
            return {}

    def load_jsonl_records(p):
        if not p.exists():
            return []
        recs = []
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    recs.append(json.loads(line))
        except Exception as exc:
            logger.warning("Failed to parse JSONL file %s: %s", p, exc)
        return recs

    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    quality = metrics.get("quality", {})
    quality_file = load_json(quality_path)
    cf_quality_file = load_json(cf_quality_path)
    rejected_row_records = load_jsonl_records(rejected_rows_path)

    thresholds = profile_thresholds(profile, expected_rows or int(manifest.get("sample_size_requested") or PROFILE_DEFAULTS.get(profile, 200)))
    expected_rows = thresholds["expected_rows"]

    failures: list[str] = []
    warnings: list[str] = []

    # Basic manifest checks
    requested = int(manifest.get("sample_size_requested", 0) or 0)
    loaded = int(manifest.get("sample_size_loaded", 0) or 0)
    if requested != expected_rows: failures.append(f"sample_size_requested is {requested}, expected {expected_rows}")
    if loaded < requested: failures.append(f"sample_size mismatch: requested {requested}, loaded {loaded}")

    # Ensure stable unique row_ids (EC-P0 fix)
    all_row_ids = set()
    for split in ["train", "val", "test"]:
        path = output_dir / f"{split}.jsonl"
        if path.exists():
            records = load_jsonl_records(path)
            for idx, r in enumerate(records):
                rid = r.get("row_id")
                if not rid or rid == "unknown":
                    failures.append(f"missing or unknown row_id in {split}.jsonl at line {idx+1}")
                    break
                if rid in all_row_ids:
                    failures.append(f"duplicate row_id found: {rid}")
                    break
                all_row_ids.add(rid)

    # Detailed metrics checks
    evidence = quality.get("evidence", {}) or {}
    canonicalization = quality.get("canonicalization", {}) or {}
    novelty = quality.get("novelty_distribution", {}) or {}
    hardness = quality.get("hardness_distribution", {}) or {}
    aspect_memory = metrics.get("aspect_memory", {}) or {}
    domain_holdout = metrics.get("domain_holdout", {}) or {}
    cf_payload = metrics.get("counterfactual_pairs", {}) or {}
    cf_stats = cf_payload.get("stats", {}) or {}

    total_exported = int(quality.get("total_exported", 0) or 0)
    full_review_rate = float(evidence.get("full_review_evidence_rate", 1.0))
    matched_term_rate = float(evidence.get("matched_term_in_evidence_rate", 1.0))
    anchor_modifier_count = int(canonicalization.get("anchor_modifier_count", 0))
    abstain_count = int(hardness.get("H2", 0) or 0) + int(hardness.get("H3", 0) or 0)
    abstain_rate = abstain_count / max(1, total_exported)
    strict_novel_rate = int(novelty.get("novel", 0) or 0) / max(1, total_exported)
    domain_holdout_val = int((domain_holdout.get("counts", {}) or {}).get("val", 0))
    counterfactual_validated = int(cf_stats.get("validated", 0) or cf_stats.get("generated", 0) or 0)
    
    cf_quality_type_counts = cf_quality_file.get("type_counts", {}) or {}
    aspect_swap_count = int(cf_quality_type_counts.get("aspect_swap", 0) or 0)
    review_queue_count = int(aspect_memory.get("review_queue_count", 0) or 0)
    unknown_candidate_count = int(aspect_memory.get("unknown_candidate_count", 0) or 0)
    broad_noun_rate = float(aspect_memory.get("broad_noun_candidate_rate", 0.0) or 0.0)
    bootstrap_entry_count = int(aspect_memory.get("bootstrap_entry_count", 0) or 0)

    # Apply thresholds
    if total_exported < thresholds["min_exported_rows"]:
        failures.append(f"exported rows is {total_exported}, expected >= {thresholds['min_exported_rows']}")
    if full_review_rate > thresholds["full_review_evidence_rate_max"]:
        failures.append(f"full_review_evidence_rate is {full_review_rate:.2%}, expected <= {thresholds['full_review_evidence_rate_max']:.2%}")
    if anchor_modifier_count < thresholds["min_anchor_modifier_count"]:
        failures.append(f"anchor_modifier_count is {anchor_modifier_count}, expected >= {thresholds['min_anchor_modifier_count']}")
    
    # These are the ones failing in the user's 800-run
    if not (thresholds["abstain_rate_min"] <= abstain_rate <= thresholds["abstain_rate_max"]):
        failures.append(f"abstain rate is {abstain_rate:.2%}, expected between {thresholds['abstain_rate_min']:.0%} and {thresholds['abstain_rate_max']:.0%}")
    if not (thresholds["novel_rate_min"] <= strict_novel_rate <= thresholds["novel_rate_max"]):
        failures.append(f"strict novel rate is {strict_novel_rate:.2%}, expected between {thresholds['novel_rate_min']:.0%} and {thresholds['novel_rate_max']:.0%}")
    
    if domain_holdout_val < thresholds["min_domain_holdout_val"]:
        failures.append(f"domain_holdout val is {domain_holdout_val}, expected >= {thresholds['min_domain_holdout_val']}")
    if counterfactual_validated < thresholds["min_counterfactual_validated"]:
        failures.append(f"counterfactual validated count is {counterfactual_validated}, expected >= {thresholds['min_counterfactual_validated']}")
    if aspect_swap_count < thresholds["min_aspect_swap_count"]:
        failures.append(f"counterfactual aspect_swap_count is {aspect_swap_count}, expected >= {thresholds['min_aspect_swap_count']}")
    if review_queue_count < thresholds["review_queue_min"] and expected_rows >= 100:
        failures.append(f"aspect_memory review_queue_count is {review_queue_count}, expected >= {thresholds['review_queue_min']}")

    quality_status = "pass" if not failures else "fail"
    artifact_status = "pass" if not failures else "fail"
    ready_for_protonet = artifact_status == "pass" and expected_rows >= 100

    report = {
        "artifact_status": artifact_status,
        "quality_status": quality_status,
        "ready_for_protonet": ready_for_protonet,
        "failed_checks": failures,
        "warnings": warnings,
        "metrics": {
            "loaded_rows": loaded,
            "exported_rows": total_exported,
            "abstain_rate": abstain_rate,
            "strict_novel_rate": strict_novel_rate,
            "counterfactual_validated": counterfactual_validated,
            "aspect_swap_count": aspect_swap_count,
            "review_queue_count": review_queue_count,
        }
    }
    return report
