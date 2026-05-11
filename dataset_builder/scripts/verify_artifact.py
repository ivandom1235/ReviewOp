from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from dataset_builder.benchmark.verification_policy import PROFILE_DEFAULTS, profile_thresholds


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    except Exception:
        return []
    return records


def verify(
    output_dir: str,
    *,
    profile: str = "development",
    expected_rows: int | None = None,
    require_organic_memory: bool = False,
    require_rejected_row_audit: bool = False,
    require_counterfactual_source_breakdown: bool = False,
) -> int:
    p = Path(output_dir)
    manifest_path = p / "manifest.json"
    metrics_path = p / "metrics_summary.json"
    quality_path = p / "quality_report.json"
    cf_quality_path = p / "counterfactual" / "counterfactual_quality_report.json"
    rejected_rows_path = p / "rejected_rows.jsonl"

    if not manifest_path.exists():
        print(f"FAILED: manifest.json not found in {output_dir}")
        return 1

    if not metrics_path.exists():
        print(f"FAILED: metrics_summary.json not found in {output_dir}")
        return 1

    manifest = _load_json(manifest_path)
    metrics = _load_json(metrics_path)
    quality = metrics.get("quality", {}) if isinstance(metrics.get("quality", {}), dict) else {}
    quality_file = _load_json(quality_path)
    cf_quality_file = _load_json(cf_quality_path)
    rejected_row_records = _load_jsonl_records(rejected_rows_path)

    thresholds = profile_thresholds(profile, expected_rows or int(manifest.get("sample_size_requested") or PROFILE_DEFAULTS.get(profile, 200)))
    expected_rows = thresholds["expected_rows"]

    failures: list[str] = []
    warnings: list[str] = []

    required_manifest_fields = (
        "run_id",
        "run_command",
        "config_hash",
        "code_hash",
        "artifact_created_at",
        "artifact_version",
        "sample_size_requested",
        "sample_size_loaded",
        "release_status",
        "gate_status",
    )
    for field in required_manifest_fields:
        if not manifest.get(field):
            failures.append(f"manifest field '{field}' is missing or empty")

    requested = int(manifest.get("sample_size_requested", 0) or 0)
    loaded = int(manifest.get("sample_size_loaded", 0) or 0)
    if requested != expected_rows:
        failures.append(f"sample_size_requested is {requested}, expected {expected_rows}")
    if loaded != expected_rows:
        failures.append(f"sample_size_loaded is {loaded}, expected {expected_rows}")
    if loaded < requested:
        failures.append(f"sample_size mismatch: requested {requested}, loaded {loaded}")

    if manifest.get("release_status") != "passed":
        failures.append(f"release_status is '{manifest.get('release_status')}', expected 'passed'")
    if str(manifest.get("gate_status", "PASS")).upper() != "PASS":
        failures.append(f"gate_status is '{manifest.get('gate_status')}', expected 'PASS'")

    required_files = [
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "manifest.json",
        "metrics_summary.json",
        "quality_report.json",
        "rejected_rows.jsonl",
        "aspect_memory_summary.json",
        "domain_holdout/manifest.json",
        "domain_holdout/train.jsonl",
        "domain_holdout/val.jsonl",
        "domain_holdout/test.jsonl",
        "counterfactual/counterfactual_pairs.jsonl",
        "counterfactual/originals.jsonl",
        "counterfactual/counterfactuals.jsonl",
        "counterfactual/counterfactual_quality_report.json",
    ]
    for rel_path in required_files:
        if not (p / rel_path).exists():
            failures.append(f"required file missing: {rel_path}")

    if quality_file and not quality:
        warnings.append("quality_report.json exists but metrics_summary quality payload is empty")

    evidence = quality.get("evidence", {}) if isinstance(quality.get("evidence", {}), dict) else {}
    canonicalization = quality.get("canonicalization", {}) if isinstance(quality.get("canonicalization", {}), dict) else {}
    novelty = quality.get("novelty_distribution", {}) if isinstance(quality.get("novelty_distribution", {}), dict) else {}
    hardness = quality.get("hardness_distribution", {}) if isinstance(quality.get("hardness_distribution", {}), dict) else {}
    aspect_memory = metrics.get("aspect_memory", {}) if isinstance(metrics.get("aspect_memory", {}), dict) else {}
    domain_holdout = metrics.get("domain_holdout", {}) if isinstance(metrics.get("domain_holdout", {}), dict) else {}
    cf_payload = metrics.get("counterfactual_pairs", {}) if isinstance(metrics.get("counterfactual_pairs", {}), dict) else {}
    cf_stats = cf_payload.get("stats", {}) if isinstance(cf_payload.get("stats", {}), dict) else {}

    total_exported = int(quality.get("total_exported", 0) or 0)
    full_review_rate = float(evidence.get("full_review_evidence_rate", 1.0) or 1.0)
    matched_term_rate = float(evidence.get("matched_term_in_evidence_rate", 1.0) or 1.0)
    anchor_modifier_count = int(canonicalization.get("anchor_modifier_count", 0) or 0)
    row_metadata_unknown_count = int(canonicalization.get("row_metadata_unknown_count", 0) or 0)
    mapping_scope_unknown_count = int(canonicalization.get("mapping_scope_unknown_count", 0) or 0)
    abstain_count = int(hardness.get("H2", 0) or 0) + int(hardness.get("H3", 0) or 0)
    abstain_rate = abstain_count / max(1, total_exported)
    strict_novel_rate = int(novelty.get("novel", 0) or 0) / max(1, total_exported)
    boundary_rate = int(novelty.get("boundary", 0) or 0) / max(1, total_exported)
    domain_holdout_val = int((domain_holdout.get("counts", {}) or {}).get("val", 0) or 0)
    counterfactual_generated = int(cf_stats.get("generated", 0) or 0)
    counterfactual_exported = int(cf_payload.get("total", 0) or 0)
    counterfactual_validated = int(cf_stats.get("validated", counterfactual_generated) or counterfactual_generated)
    counterfactual_rejected_unnatural = int(cf_stats.get("rejected_unnatural", 0) or 0)
    counterfactual_rejected_no_expected_change = int(cf_stats.get("rejected_no_expected_change", max(0, int(cf_stats.get("attempted", 0) or 0) - counterfactual_generated)) or 0)
    cf_quality_attempted = int(cf_quality_file.get("attempted", 0) or 0)
    cf_quality_generated = int(cf_quality_file.get("generated", 0) or 0)
    cf_quality_exported = int(cf_quality_file.get("exported", 0) or 0)
    cf_quality_validated = int(cf_quality_file.get("validated", 0) or 0)
    cf_quality_rejected_unnatural = int(cf_quality_file.get("rejected_unnatural", 0) or 0)
    cf_quality_rejected_no_expected_change = int(cf_quality_file.get("rejected_no_expected_change", 0) or 0)
    cf_quality_type_counts = cf_quality_file.get("type_counts", {}) if isinstance(cf_quality_file.get("type_counts", {}), dict) else {}
    aspect_swap_count = int(cf_quality_type_counts.get("aspect_swap", 0) or 0)
    review_queue_count = int(aspect_memory.get("review_queue_count", 0) or 0)
    promoted_count = int(aspect_memory.get("promoted_entries_total", 0) or aspect_memory.get("promoted_count", 0) or 0)
    unknown_candidate_count = int(aspect_memory.get("unknown_candidate_count", 0) or 0)
    broad_noun_rate = float(aspect_memory.get("broad_noun_candidate_rate", 0.0) or 0.0)
    bootstrap_entry_count = int(aspect_memory.get("bootstrap_entry_count", 0) or 0)
    organic_review_queue_count = int(aspect_memory.get("organic_review_queue_count", review_queue_count) or 0)
    rejected_rows_count = int(quality_file.get("rejected_rows", quality.get("rejected_rows", 0)) or 0)
    cf_source_counts = cf_stats.get("source_counts", {}) if isinstance(cf_stats.get("source_counts", {}), dict) else {}
    natural_aspect_swap_count = int(cf_stats.get("natural_aspect_swap_count", 0) or 0)
    synthetic_aspect_swap_count = int(cf_stats.get("synthetic_aspect_swap_count", 0) or 0)

    if total_exported < thresholds["min_exported_rows"]:
        failures.append(f"exported rows is {total_exported}, expected >= {thresholds['min_exported_rows']}")
    if loaded != expected_rows:
        failures.append(f"loaded rows is {loaded}, expected {expected_rows}")
    if full_review_rate > thresholds["full_review_evidence_rate_max"]:
        failures.append(f"full_review_evidence_rate is {full_review_rate:.2%}, expected <= {thresholds['full_review_evidence_rate_max']:.2%}")
    if anchor_modifier_count < thresholds["min_anchor_modifier_count"]:
        failures.append(f"anchor_modifier_count is {anchor_modifier_count}, expected >= {thresholds['min_anchor_modifier_count']}")
    if not (thresholds["abstain_rate_min"] <= abstain_rate <= thresholds["abstain_rate_max"]):
        failures.append(f"abstain rate is {abstain_rate:.2%}, expected between {thresholds['abstain_rate_min']:.0%} and {thresholds['abstain_rate_max']:.0%}")
    if not (thresholds["novel_rate_min"] <= strict_novel_rate <= thresholds["novel_rate_max"]):
        failures.append(f"strict novel rate is {strict_novel_rate:.2%}, expected between {thresholds['novel_rate_min']:.0%} and {thresholds['novel_rate_max']:.0%}")
    if domain_holdout_val < thresholds["min_domain_holdout_val"]:
        failures.append(f"domain_holdout val is {domain_holdout_val}, expected >= {thresholds['min_domain_holdout_val']}")
    if counterfactual_validated < thresholds["min_counterfactual_validated"]:
        failures.append(f"counterfactual validated count is {counterfactual_validated}, expected >= {thresholds['min_counterfactual_validated']}")
    if counterfactual_exported < counterfactual_validated:
        failures.append(f"counterfactual exported count is {counterfactual_exported}, validated count is {counterfactual_validated}")
    required_cf_quality_fields = ("attempted", "generated", "exported", "validated", "rejected_unnatural", "rejected_no_expected_change", "type_counts")
    missing_cf_quality_fields = [field for field in required_cf_quality_fields if field not in cf_quality_file]
    if not cf_quality_file:
        failures.append("counterfactual quality report is missing or empty")
    elif missing_cf_quality_fields:
        failures.append(f"counterfactual quality report is missing fields: {', '.join(missing_cf_quality_fields)}")
    if cf_quality_file and not cf_quality_type_counts:
        failures.append("counterfactual quality report type_counts is empty")
    if cf_quality_file and cf_quality_validated < thresholds["min_counterfactual_validated"]:
        failures.append(f"counterfactual quality report validated count is {cf_quality_validated}, expected >= {thresholds['min_counterfactual_validated']}")
    if cf_quality_file and cf_quality_exported < cf_quality_validated:
        failures.append(f"counterfactual quality report exported count is {cf_quality_exported}, validated count is {cf_quality_validated}")
    if cf_quality_file and cf_quality_generated < cf_quality_validated:
        failures.append(f"counterfactual quality report generated count is {cf_quality_generated}, validated count is {cf_quality_validated}")
    if cf_quality_file and cf_quality_attempted < cf_quality_generated:
        failures.append(f"counterfactual quality report attempted count is {cf_quality_attempted}, generated count is {cf_quality_generated}")
    if cf_quality_file and cf_quality_rejected_unnatural == 0 and cf_quality_rejected_no_expected_change == 0:
        warnings.append("counterfactual quality report rejection breakdown is empty")
    if aspect_swap_count < thresholds["min_aspect_swap_count"]:
        failures.append(f"counterfactual aspect_swap_count is {aspect_swap_count}, expected >= {thresholds['min_aspect_swap_count']}")
    if review_queue_count < thresholds["review_queue_min"] and expected_rows >= 100:
        failures.append(f"aspect_memory review_queue_count is {review_queue_count}, expected >= {thresholds['review_queue_min']}")
    if unknown_candidate_count > thresholds["unknown_candidate_count_max"]:
        failures.append(f"unknown_candidate_count is {unknown_candidate_count}, expected {thresholds['unknown_candidate_count_max']}")
    if broad_noun_rate > thresholds["broad_noun_rate_max"]:
        failures.append(f"broad_noun_candidate_rate is {broad_noun_rate:.2%}, expected <= {thresholds['broad_noun_rate_max']:.2%}")
    if mapping_scope_unknown_count > 0:
        failures.append(f"found {mapping_scope_unknown_count} rows with unknown mapping_scope")
    if row_metadata_unknown_count > 0:
        failures.append(f"found {row_metadata_unknown_count} rows with unknown row metadata")
    if require_organic_memory:
        if bootstrap_entry_count > 0:
            failures.append(f"bootstrap_entry_count is {bootstrap_entry_count}, expected 0 for organic-memory verification")
        if organic_review_queue_count < thresholds["review_queue_min"]:
            failures.append(f"organic_review_queue_count is {organic_review_queue_count}, expected >= {thresholds['review_queue_min']}")
        if broad_noun_rate > thresholds["broad_noun_rate_max"]:
            failures.append(f"organic memory broad_noun_candidate_rate is {broad_noun_rate:.2%}, expected <= {thresholds['broad_noun_rate_max']:.2%}")
        if unknown_candidate_count > thresholds["unknown_candidate_count_max"]:
            failures.append(f"organic memory unknown_candidate_count is {unknown_candidate_count}, expected {thresholds['unknown_candidate_count_max']}")
    if require_rejected_row_audit:
        if rejected_rows_count != len(rejected_row_records):
            failures.append(f"rejected_rows.jsonl contains {len(rejected_row_records)} rows, but quality_report rejected_rows is {rejected_rows_count}")
        for idx, record in enumerate(rejected_row_records, start=1):
            if "candidate_trace" not in record:
                failures.append(f"rejected_rows.jsonl row {idx} is missing candidate_trace")
            if not record.get("recommended_recovery"):
                failures.append(f"rejected_rows.jsonl row {idx} is missing recommended_recovery")
    if require_counterfactual_source_breakdown:
        if not cf_source_counts:
            failures.append("counterfactual source_counts is missing or empty")
        if natural_aspect_swap_count <= 0:
            failures.append("counterfactual natural_aspect_swap_count is missing or zero")
        if synthetic_aspect_swap_count < 0:
            failures.append("counterfactual synthetic_aspect_swap_count is invalid")
    if matched_term_rate < 0.97:
        warnings.append(f"matched_term_in_evidence_rate is {matched_term_rate:.2%}")
    if counterfactual_rejected_unnatural == 0 and counterfactual_rejected_no_expected_change == 0:
        warnings.append("counterfactual rejection breakdown is empty")

    quality_status = "pass" if not failures else "fail"
    source_artifact_consistency = {
        "code_hash": manifest.get("code_hash", ""),
        "config_hash": manifest.get("config_hash", ""),
        "run_command": manifest.get("run_command", ""),
        "sample_size_requested": requested,
        "sample_size_loaded": loaded,
        "artifact_matches_source": requested == expected_rows and loaded == expected_rows,
    }
    (p / "source_artifact_consistency.json").write_text(json.dumps(source_artifact_consistency, indent=2), encoding="utf-8")

    artifact_status = "pass" if not failures else "fail"
    ready_for_protonet = artifact_status == "pass" and expected_rows >= 100 and review_queue_count >= thresholds["review_queue_min"]

    verification_report = {
        "artifact_status": artifact_status,
        "quality_status": quality_status,
        "ready_for_protonet": ready_for_protonet,
        "failed_checks": failures,
        "warnings": warnings,
        "metrics": {
            "loaded_rows": loaded,
            "exported_rows": total_exported,
            "rejection_rate": max(0.0, 1.0 - (total_exported / max(1, loaded))),
            "domain_holdout_val": domain_holdout_val,
            "counterfactual_validated": counterfactual_validated,
            "aspect_memory_review_queue": review_queue_count,
            "anchor_modifier_count": anchor_modifier_count,
            "full_review_evidence_rate": full_review_rate,
            "abstain_rate": abstain_rate,
            "strict_novel_rate": strict_novel_rate,
            "boundary_rate": boundary_rate,
            "bootstrap_entry_count": bootstrap_entry_count,
            "organic_review_queue_count": organic_review_queue_count,
            "rejected_rows_audit_count": len(rejected_row_records),
            "natural_aspect_swap_count": natural_aspect_swap_count,
            "synthetic_aspect_swap_count": synthetic_aspect_swap_count,
        },
        "source_artifact_consistency": source_artifact_consistency,
    }

    (p / "artifact_verification.json").write_text(json.dumps(verification_report, indent=2), encoding="utf-8")
    artifact_zip_path = p / "artifact.zip"
    if artifact_zip_path.exists():
        with ZipFile(artifact_zip_path, "a", compression=ZIP_DEFLATED) as archive:
            archive.write(p / "artifact_verification.json", arcname="artifact_verification.json")
            archive.write(p / "source_artifact_consistency.json", arcname="source_artifact_consistency.json")

    if failures:
        print("VERIFICATION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("ARTIFACT VERIFIED SUCCESSFULLY")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a dataset_builder artifact directory.")
    parser.add_argument("output_dir")
    parser.add_argument("--profile", default="development", choices=sorted(PROFILE_DEFAULTS))
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--require-organic-memory", action="store_true")
    parser.add_argument("--require-rejected-row-audit", action="store_true")
    parser.add_argument("--require-counterfactual-source-breakdown", action="store_true")
    args = parser.parse_args(argv)
    return verify(
        args.output_dir,
        profile=args.profile,
        expected_rows=args.expected_rows,
        require_organic_memory=args.require_organic_memory,
        require_rejected_row_audit=args.require_rejected_row_audit,
        require_counterfactual_source_breakdown=args.require_counterfactual_source_breakdown,
    )


if __name__ == "__main__":
    sys.exit(main())
