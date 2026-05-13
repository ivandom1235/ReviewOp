from __future__ import annotations

import json
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil
from collections import defaultdict

from rich.progress import Progress

from ..config import BuilderConfig
from ..export.archive import write_artifact_zip
from ..export.jsonl_export import write_jsonl_rows, write_split_jsonl
from ..export.manifest import write_manifest
from ..export.sidecars import write_sidecar
from ..benchmark.counterfactual import generate_counterfactual_pairs
from ..reports.quality_report import build_quality_report
from ..utils.row_metadata import derive_row_metadata
from ..schemas.artifact_manifest import ArtifactManifest
from ..split.leakage_checks import check_cross_split_leakage
from ..split.domain_split import choose_domain_holdout_domain, domain_holdout_split
from .release_gate import assert_release_ready
from .exceptions import QualityGateError
from .stages import (
    ExtractionStage,
    InferenceStage,
    EvidenceStage,
    VerificationStage,
    PostVerificationEvidenceStage,
    CanonicalizationStage,
    FusionStage,
    SentimentStage,
    BenchmarkStage,
)
from ..schemas.benchmark_row import BenchmarkRow
from ..schemas.raw_review import RawReview
from ..split.grouped_split import grouped_train_val_test_split
from ..canonical.domain_registry import DomainRegistry
logger = logging.getLogger(__name__)


def _remove_near_duplicates(rows: list[BenchmarkRow], threshold: float = 0.95) -> list[BenchmarkRow]:
    """Remove reviews with near-duplicate text to prevent cross-split leakage."""
    from rapidfuzz import fuzz
    unique_rows: list[BenchmarkRow] = []
    seen_texts: list[str] = []
    
    for row in rows:
        text = row.review_text.strip().lower()
        if not text:
            unique_rows.append(row)
            continue
            
        is_duplicate = False
        # Only check against last N for speed if dataset is huge, but for 1000 it's fine
        for other_text in seen_texts:
            if fuzz.ratio(text, other_text) / 100.0 >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_rows.append(row)
            seen_texts.append(text)
            
    return unique_rows


def _get_code_hash() -> str:
    """Simple hash of the dataset_builder package to ensure artifact-source consistency."""
    h = hashlib.sha256()
    try:
        package_dir = Path(__file__).resolve().parents[1]
        for p in sorted(package_dir.rglob("*.py")):
            if "__pycache__" in str(p) or "venv" in str(p) or ".venv" in str(p):
                continue
            with open(p, "rb") as f:
                h.update(f.read())
    except Exception:
        h.update(b"unknown")
    return h.hexdigest()[:12]


def _load_json_dict(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _calculate_checksums(output_dir: Path) -> dict[str, str]:
    """Calculate SHA256 checksums for core artifact files."""
    files = [
        "train.jsonl", "val.jsonl", "test.jsonl",
        "metrics_summary.json", "quality_report.json",
        "rejected_rows.jsonl",
        "counterfactual_pairs.json",
        "counterfactual_pairs.jsonl",
        "aspect_memory_summary.json"
    ]
    checksums = {}
    for rel_path in files:
        p = output_dir / rel_path
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read())
            checksums[rel_path] = f"sha256:{h.hexdigest()}"
    return checksums


def _build_label_equivalence(rows: list[BenchmarkRow]) -> dict[str, list[str]]:
    eq: dict[str, set[str]] = defaultdict(set)
    domains = sorted({str(r.domain or "generic").lower() for r in rows})
    for domain in domains:
        cfg = DomainRegistry.get_config(domain)
        domain_map = cfg.get("domain_maps", {}) if isinstance(cfg, dict) else {}
        for alias, canonical in (domain_map or {}).items():
            a = str(alias or "").strip().lower().replace(" ", "_")
            c = str(canonical or "").strip().lower().replace(" ", "_")
            if not a or not c:
                continue
            eq[c].add(a)
            eq[c].add(c)
        canonical_aliases = cfg.get("canonical_aliases", {}) if isinstance(cfg, dict) else {}
        for canonical, aliases in (canonical_aliases or {}).items():
            c = str(canonical or "").strip().lower().replace(" ", "_")
            if not c:
                continue
            eq[c].add(c)
            for alias in aliases or []:
                a = str(alias or "").strip().lower().replace(" ", "_")
                if a:
                    eq[c].add(a)
    return {k: sorted(v) for k, v in sorted(eq.items())}


def _build_promoted_memory_sidecar(memory_path: str | None) -> dict[str, object]:
    if not memory_path:
        return {"created_at": datetime.now(timezone.utc).isoformat(), "total": 0, "items": []}
    p = Path(memory_path)
    if not p.exists():
        return {"created_at": datetime.now(timezone.utc).isoformat(), "total": 0, "items": []}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        items = []
        for cluster_id, entry in entries.items():
            if str(entry.get("status", "")).strip().lower() != "promoted":
                continue
            items.append(
                {
                    "cluster_id": cluster_id,
                    "aspect_raw": entry.get("aspect_raw"),
                    "suggested_aspect": entry.get("suggested_aspect"),
                    "support_count": int(entry.get("support_count", 0) or 0),
                    "trigger_patterns": list(entry.get("trigger_patterns", []) or []),
                    "quality": float(entry.get("evidence_quality_mean", 0.0) or 0.0),
                    "consistency": float(entry.get("cluster_consistency", 0.0) or 0.0),
                    "status": "promoted",
                }
            )
        return {"created_at": datetime.now(timezone.utc).isoformat(), "total": len(items), "items": items}
    except Exception:
        return {"created_at": datetime.now(timezone.utc).isoformat(), "total": 0, "items": []}


def run_builder_pipeline(
    cfg: BuilderConfig, 
    raw_reviews: list[RawReview] | None = None,
    rows_by_split: dict[str, list[BenchmarkRow]] | None = None, 
    profile_summary: dict[str, object] | None = None,
    original_sample_size: int | None = None,
) -> dict[str, object]:
    """
    Main entry point for the builder pipeline.
    If raw_reviews are provided, it runs Stages A-F and then splits.
    If rows_by_split are provided directly, it skips to checks and export.
    """
    output_dir = Path(cfg.output_dir)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output cleaning (Requirement)
    if not cfg.dry_run:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not cfg.overwrite:
                raise FileExistsError(f"output directory is not empty: {output_dir} (use --overwrite to clear)")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.aspect_memory_path:
        cfg = BuilderConfig(**{**cfg.__dict__, "aspect_memory_path": str(output_dir / "aspect_memory_candidates.json")})
    
    if raw_reviews is not None:
        requested_rows = cfg.sample_size if cfg.sample_size is not None else 0
        loaded_rows = len(raw_reviews)
        
        # Phase 1: Sample size verification
        if requested_rows > 0 and loaded_rows < requested_rows:
            import logging
            logging.getLogger("dataset_builder").error(
                f"Sample size mismatch: Requested {requested_rows}, but only loaded {loaded_rows} rows."
            )
            # We don't raise here yet, but we'll mark the release_status as failed later
        
        # Step 1: Initial Conversion
        rows = [
            BenchmarkRow(
                review_id=r.review_id,
                group_id=r.group_id,
                domain=r.domain,
                domain_family=r.domain_family,
                review_text=r.text,
                gold_interpretations=[], # Will be filled by stages
                provenance={
                    "source_name": r.source_name,
                    "source_split": r.source_split,
                    "metadata": dict(r.metadata or {}),
                }
            ) for r in raw_reviews
        ]
        
        # Step 2: Run Stages A-F
        from rich.progress import TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn
        from .telemetry import GLOBAL_STATS
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[stats]}"),
        ) as progress:
            t_stages = progress.add_task("[cyan]Building Benchmark...", total=9, stats="")
            
            stages = [
                ExtractionStage(),
                InferenceStage(),
                FusionStage(),
                EvidenceStage(),
                VerificationStage(),
                PostVerificationEvidenceStage(),
                CanonicalizationStage(),
                SentimentStage(),
                BenchmarkStage(),
            ]
            
            import threading
            import time
            
            stop_event = threading.Event()
            
            def update_stats():
                while not stop_event.is_set():
                    stats_str = (
                        f"LLM: {GLOBAL_STATS.llm_calls} | "
                        f"Cache: {GLOBAL_STATS.cached_llm_calls} | "
                        f"Fallback: {GLOBAL_STATS.fallback_calls}"
                    )
                    row_progress = ""
                    if GLOBAL_STATS.current_stage_total > 0:
                        row_progress = f" | Rows: {GLOBAL_STATS.current_stage_processed}/{GLOBAL_STATS.current_stage_total}"
                    
                    progress.update(t_stages, stats=f"{stats_str}{row_progress}")
                    time.sleep(0.5)
            
            updater_thread = threading.Thread(target=update_stats, daemon=True)
            updater_thread.start()
            
            try:
                for stage in stages:
                    stage_name = stage.__class__.__name__
                    progress.update(t_stages, description=f"[cyan]Running {stage_name}...")
                    rows = stage.process(rows, cfg)
                    progress.update(t_stages, advance=1)
            finally:
                stop_event.set()
                updater_thread.join(timeout=1.0)
                
        processed_rows = len(rows)
        rejected_rows = loaded_rows - processed_rows
        discarded_rows = 0 # Future expansion
        
        # Step 2.5: Near-Duplicate Removal (Requirement: Leakage-free splits)
        # We do this before splitting to ensure no near-duplicates end up in different splits
        pre_dedup_count = len(rows)
        rows = _remove_near_duplicates(rows, threshold=0.95)
        dedup_dropped = pre_dedup_count - len(rows)
        if dedup_dropped > 0:
            import logging
            logging.getLogger("dataset_builder").info(f"Dropped {dedup_dropped} near-duplicate reviews to ensure zero leakage.")
        
        # Step 3: Split
        if cfg.domain_holdout_domain:
            holdout_domain = cfg.domain_holdout_domain
            domain_splits = domain_holdout_split(rows, holdout_domain=holdout_domain)
            non_holdout_rows = domain_splits["train"] + domain_splits["val"]
            if non_holdout_rows:
                total_tv = cfg.train_ratio + cfg.val_ratio
                internal_train_ratio = cfg.train_ratio / total_tv if total_tv > 0 else 0.8
                internal_val_ratio = cfg.val_ratio / total_tv if total_tv > 0 else 0.2
                tv_splits = grouped_train_val_test_split(
                    non_holdout_rows,
                    seed=cfg.random_seed,
                    train_ratio=internal_train_ratio,
                    val_ratio=internal_val_ratio,
                    test_ratio=0.0
                )
                rows_by_split = {"train": tv_splits["train"], "val": tv_splits["val"], "test": domain_splits["test"]}
            else:
                rows_by_split = domain_splits
        else:
            rows_by_split = grouped_train_val_test_split(
                rows,
                seed=cfg.random_seed,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
            )
        rows_by_split = {
            split: [derive_row_metadata(row) for row in split_rows]
            for split, split_rows in rows_by_split.items()
        }
    else:
        # If provided directly via rows_by_split
        requested_rows = sum(len(s) for s in rows_by_split.values())
        loaded_rows = requested_rows
        processed_rows = requested_rows
        rejected_rows = 0
        discarded_rows = 0

    if rows_by_split is None:
        raise ValueError("Either raw_reviews or rows_by_split must be provided")

    # Ensure all rows have a stable unique row_id
    from dataclasses import replace
    rows_by_split = {
        split: [
            replace(row, row_id=f"{run_id}_{split}_{idx:06d}")
            for idx, row in enumerate(split_rows)
        ]
        for split, split_rows in rows_by_split.items()
    }

    rejected_audit = list(getattr(cfg, "_rejected_rows_audit", []) or [])
    if rejected_audit:
        rejected_rows = max(rejected_rows, len(rejected_audit))

    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    domain_holdout_domain = (cfg.domain_holdout_domain or choose_domain_holdout_domain(all_rows)) if all_rows else (cfg.domain_holdout_domain or "unknown")
    domain_holdout_rows = domain_holdout_split(all_rows, domain_holdout_domain) if all_rows else {"train": [], "val": [], "test": []}
    cf_res = generate_counterfactual_pairs(all_rows)
    counterfactual_pairs = cf_res["pairs"]
    counterfactual_stats = cf_res["stats"]

    # Reproducibility metadata
    run_command = " ".join(sys.argv)
    code_hash = _get_code_hash()
    config_hash = hashlib.sha256(str(cfg.__dict__).encode()).hexdigest()[:12]
    
    source_consistency = {
        "code_hash": code_hash,
        "config_hash": config_hash,
        "run_command": run_command,
        "sample_size_requested": requested_rows,
        "sample_size_loaded": loaded_rows,
        "artifact_matches_source": True # Always true for new runs, but checked if re-running
    }

    with Progress() as progress:
        t1 = progress.add_task("[green]Quality & Leakage Checks...", total=3)
        quality = build_quality_report(
            rows_by_split, 
            requested_rows=requested_rows,
            loaded_rows=loaded_rows,
            processed_rows=processed_rows,
            rejected_rows=rejected_rows,
            discarded_rows=discarded_rows,
            runtime_reason_counts=getattr(cfg, "_rejection_reason_counts", {}) or {},
            original_sample_size=original_sample_size or loaded_rows,
            source_consistency=source_consistency,
        )
        progress.update(t1, advance=1)
        leakage_results = check_cross_split_leakage(rows_by_split)
        progress.update(t1, advance=2)
        
        leakage = {
            "grouped_leakage": int(leakage_results["grouped_leakage"]),
            "exact_text_leakage": int(leakage_results["exact_text_leakage"]),
            "near_duplicate_leakage": int(leakage_results.get("near_duplicate_leakage", 0)),
        }
        
        profile = "diagnostic_strict" if getattr(cfg, "strict", False) else getattr(cfg, "profile", "development")
        metrics = {
            "counts": quality.export_counts,
            "quality": quality.__dict__ if hasattr(quality, "__dict__") else quality,
            "leakage": leakage,
            "profile": profile,
            "domain_holdout": {
                "domain": domain_holdout_domain,
                "counts": {split: len(rows) for split, rows in domain_holdout_rows.items()},
            },
            "counterfactual_pairs": {
                "total": len(counterfactual_pairs),
                "stats": counterfactual_stats
            },
        }
        aspect_memory_metrics = {
            "candidates_added": 0,
            "promoted_matches_used": 0,
            "candidates_promoted_this_run": 0,
            "promoted_entries_total": 0,
            "review_queue_count": 0,
            "rejected_candidates_this_run": 0,
        }
        aspect_memory_summary = _load_json_dict(output_dir / "aspect_memory_summary.json")
        if cfg.aspect_memory_path:
            runtime_metrics = getattr(cfg, "_aspect_memory_metrics", {}) or {}
            try:
                from ..canonical.aspect_memory import AspectMemory
                memory = AspectMemory(cfg.aspect_memory_path)
                promoted_total = sum(1 for e in memory.entries.values() if e.status == "promoted")
                review_queue_total = sum(1 for e in memory.entries.values() if e.status == "review_queue")
                aspect_memory_metrics["promoted_entries_total"] = promoted_total
                aspect_memory_metrics["review_queue_count"] = review_queue_total
            except Exception:
                logger.warning(
                    "Failed to load aspect memory metrics from %s",
                    cfg.aspect_memory_path,
                    exc_info=True,
                )
            for key in ("candidates_added", "promoted_matches_used", "candidates_promoted_this_run", "rejected_candidates_this_run"):
                if key in runtime_metrics:
                    aspect_memory_metrics[key] = runtime_metrics[key]
        if aspect_memory_summary:
            if "promoted_count" in aspect_memory_summary and "promoted_entries_total" not in aspect_memory_summary:
                aspect_memory_metrics["promoted_entries_total"] = int(aspect_memory_summary.get("promoted_count", 0) or 0)
            for key, value in aspect_memory_summary.items():
                aspect_memory_metrics[key] = value
            if "promoted_count" in aspect_memory_summary:
                aspect_memory_metrics["promoted_entries_total"] = int(aspect_memory_summary.get("promoted_count", 0) or 0)
        metrics["aspect_memory"] = aspect_memory_metrics
        metrics["anchor_modifier_debug"] = getattr(cfg, "_anchor_modifier_debug", {}) or {}
        
        try:
            gate_results = assert_release_ready(rows_by_split, reports={"quality": quality}, leakage=leakage, profile=profile)
            status_map = {"PASS": "passed", "WARNING": "warning", "FAIL": "failed", "FATAL": "failed"}
            release_status = status_map.get(gate_results.get("status"), "unknown")
            gate_status = str(gate_results.get("status", "UNKNOWN")).upper()
        except QualityGateError as e:
            gate_results = e.gate_results
            release_status = "failed"
            gate_status = str(gate_results.get("status", "UNKNOWN")).upper()
        
        # Hard fail if sample size mismatch
        if (requested_rows or 0) > 0 and loaded_rows < (requested_rows or 0):
            release_status = "failed"
            import logging
            logging.getLogger("dataset_builder").error("Release FAILED due to sample_size mismatch.")

        metrics["gate_results"] = gate_results
        
        # Mandatory export of metrics_summary.json
        if not cfg.dry_run:
            with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
                def d_ser(obj):
                    if hasattr(obj, "to_dict"): return obj.to_dict()
                    if hasattr(obj, "__dict__"): return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
                    return str(obj)
                json.dump(metrics, f, indent=2, default=d_ser)
        
        if cfg.dry_run:
            return {
                "counts": quality.export_counts,
                "quality": quality,
                "leakage": leakage,
                "domain_holdout": {"domain": domain_holdout_domain, "counts": {split: len(rows) for split, rows in domain_holdout_rows.items()}},
                "counterfactual_pairs": len(counterfactual_pairs),
                "dry_run": True,
            }
            
        t2 = progress.add_task("[yellow]Exporting Artifacts...", total=6)
        grouped_dir = output_dir / "grouped"
        write_split_jsonl(grouped_dir, rows_by_split)
        progress.update(t2, advance=1)

        counts = write_split_jsonl(output_dir, rows_by_split)
        progress.update(t2, advance=1)

        domain_holdout_dir = output_dir / "domain_holdout"
        write_split_jsonl(domain_holdout_dir, domain_holdout_rows)
        write_sidecar(
            domain_holdout_dir / "manifest.json",
            {
                "artifact_type": "domain_holdout_release",
                "domain": domain_holdout_domain,
                "counts": {split: len(rows) for split, rows in domain_holdout_rows.items()},
                "source_rows": len(all_rows),
                "protocol": "domain_holdout",
            },
        )
        counterfactual_dir = output_dir / "counterfactual"
        if counterfactual_pairs:
            write_sidecar(output_dir / "counterfactual_pairs.json", counterfactual_pairs)
            write_jsonl_rows(counterfactual_dir / "counterfactual_pairs.jsonl", counterfactual_pairs)
            write_jsonl_rows(
                counterfactual_dir / "originals.jsonl",
                [
                    {
                        "counterfactual_group_id": pair["counterfactual_group_id"],
                        "review_id": pair["original_review_id"],
                        "group_id": pair["original_group_id"],
                        "domain": pair["domain"],
                        "review_text": pair["original_text"],
                        "role": "original",
                        "split_protocol": pair.get("source_split_protocol", {}),
                    }
                    for pair in counterfactual_pairs
                ],
            )
            write_jsonl_rows(
                counterfactual_dir / "counterfactuals.jsonl",
                [
                    {
                        "counterfactual_group_id": pair["counterfactual_group_id"],
                        "review_id": pair["counterfactual_review_id"],
                        "group_id": pair["original_group_id"],
                        "domain": pair["domain"],
                        "review_text": pair["counterfactual_text"],
                        "role": "counterfactual",
                        "changed_trigger": pair["changed_trigger"],
                        "expected_behavior": pair["expected_behavior"],
                        "split_protocol": pair.get("source_split_protocol", {}),
                    }
                    for pair in counterfactual_pairs
                ],
            )
            write_sidecar(counterfactual_dir / "counterfactual_quality_report.json", counterfactual_stats)
        progress.update(t2, advance=1)

        if rejected_audit:
            write_jsonl_rows(output_dir / "rejected_rows.jsonl", rejected_audit)
        else:
            (output_dir / "rejected_rows.jsonl").write_text("", encoding="utf-8")

        write_sidecar(output_dir / "label_equivalence.json", _build_label_equivalence(all_rows))
        write_sidecar(output_dir / "aspect_memory_promoted.json", _build_promoted_memory_sidecar(cfg.aspect_memory_path))

        # Write consistency sidecar
        write_sidecar(output_dir / "source_artifact_consistency.json", source_consistency)

        # Write quality report before manifest so it can be checksummed
        write_sidecar(output_dir / "quality_report.json", quality)
        progress.update(t2, advance=1)

        # Calculate checksums for core files
        artifact_checksums = _calculate_checksums(output_dir)

        manifest = ArtifactManifest(
            version="dataset_builder_p0",
            dataset_inputs=[str(path) for path in cfg.input_paths],
            profile_summary=profile_summary or {},
            policies_used={
                "release_gate": profile,
                "llm_provider": cfg.llm_provider,
                "llm_model": cfg.llm_model,
                "random_seed": cfg.random_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "sample_size": cfg.sample_size,
                "chunk_size": cfg.chunk_size,
                "chunk_offset": cfg.chunk_offset,
                "strict": cfg.strict,
                "domain_mode": cfg.domain_mode,
                "provisional_policy": cfg.provisional_policy,
                "evidence_window_tokens": cfg.evidence_window_tokens,
                "aspect_memory_path": cfg.aspect_memory_path,
                "aspect_memory_auto_promote": cfg.aspect_memory_auto_promote,
                "aspect_memory_bootstrap": cfg.aspect_memory_bootstrap,
                "symptom_store_path": cfg.symptom_store_path,
                "domain_holdout_domain": domain_holdout_domain,
                "max_workers": cfg.max_workers,
            },
            split_summary=counts,
            release_status=release_status,
            gate_status=gate_status,
            run_id=run_id,
            run_command=run_command,
            code_hash=code_hash,
            config_hash=config_hash,
            artifact_created_at=datetime.now(timezone.utc).isoformat(),
            sample_size_requested=requested_rows,
            sample_size_loaded=loaded_rows,
            original_sample_size=original_sample_size or loaded_rows,
            artifact_checksums=artifact_checksums,
        )
        write_manifest(output_dir / "manifest.json", manifest)

        # Final strict verification check (EC-P6 fix)
        from ..benchmark.verifier import verify_artifact_dir
        v_report = verify_artifact_dir(output_dir, profile=profile, expected_rows=requested_rows)
        
        # Always write detailed verification report to artifact dir
        with open(output_dir / "artifact_verification.json", "w", encoding="utf-8") as vf:
            json.dump(v_report, vf, indent=2)

        # Override gate status based on strict verification
        if v_report["artifact_status"] == "fail":
            from dataclasses import replace
            manifest = replace(manifest, release_status="failed", gate_status="FAIL")
            write_manifest(output_dir / "manifest.json", manifest)
        progress.update(t2, advance=1)
        
        archive_path = write_artifact_zip(output_dir)
        progress.update(t2, advance=1)
        
    return {"counts": counts, "quality": quality, "leakage": leakage, "archive_path": archive_path}
