from __future__ import annotations

from pathlib import Path
import shutil
import json

from rich.progress import Progress

from ..config import BuilderConfig
from ..export.archive import write_artifact_zip
from ..export.jsonl_export import write_split_jsonl
from ..export.manifest import write_manifest
from ..export.sidecars import write_sidecar
from ..reports.quality_report import build_quality_report
from ..schemas.artifact_manifest import ArtifactManifest
from ..split.leakage_checks import check_group_leakage, check_text_duplication
from .release_gate import assert_release_ready
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


def run_builder_pipeline(
    cfg: BuilderConfig, 
    raw_reviews: list[RawReview] | None = None,
    rows_by_split: dict[str, list[BenchmarkRow]] | None = None, 
    profile_summary: dict[str, object] | None = None
) -> dict[str, object]:
    """
    Main entry point for the builder pipeline.
    If raw_reviews are provided, it runs Stages A-F and then splits.
    If rows_by_split are provided directly, it skips to checks and export.
    """
    output_dir = Path(cfg.output_dir)
    if not cfg.dry_run:
        if output_dir.exists() and any(output_dir.iterdir()):
            if not cfg.overwrite:
                raise FileExistsError(f"output directory is not empty: {output_dir} (use --overwrite to clear)")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    if raw_reviews is not None:
        requested_rows = cfg.sample_size
        loaded_rows = len(raw_reviews)
        
        # Step 1: Initial Conversion
        rows = [
            BenchmarkRow(
                review_id=r.review_id,
                group_id=r.group_id,
                domain=r.domain,
                domain_family=r.domain_family,
                review_text=r.text,
                gold_interpretations=[], # Will be filled by stages
                provenance={"source_name": r.source_name, "source_split": r.source_split}
            ) for r in raw_reviews
        ]
        
        # Step 2: Run Stages A-F
        with Progress() as progress:
            t_stages = progress.add_task("[cyan]Building Benchmark...", total=9)
            
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
            
            for stage in stages:
                rows = stage.process(rows, cfg)
                progress.update(t_stages, advance=1)
                
        processed_rows = len(rows)
        rejected_rows = loaded_rows - processed_rows
        discarded_rows = 0 # Future expansion
        
        # Step 3: Split
        rows_by_split = grouped_train_val_test_split(
            rows,
            seed=cfg.random_seed,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )
    else:
        # If provided directly via rows_by_split
        requested_rows = sum(len(s) for s in rows_by_split.values())
        loaded_rows = requested_rows
        processed_rows = requested_rows
        rejected_rows = 0
        discarded_rows = 0

    if rows_by_split is None:
        raise ValueError("Either raw_reviews or rows_by_split must be provided")

    with Progress() as progress:
        t1 = progress.add_task("[green]Quality & Leakage Checks...", total=3)
        quality = build_quality_report(
            rows_by_split, 
            requested_rows=requested_rows,
            loaded_rows=loaded_rows,
            processed_rows=processed_rows,
            rejected_rows=rejected_rows,
            discarded_rows=discarded_rows
        )
        progress.update(t1, advance=1)
        group_leakage = check_group_leakage(rows_by_split)
        progress.update(t1, advance=1)
        text_leakage = check_text_duplication(rows_by_split)
        progress.update(t1, advance=1)
        
        leakage = {
            "grouped_leakage": int(group_leakage["grouped_leakage"]),
            "exact_text_leakage": int(text_leakage["exact_text_leakage"]),
        }
        
        profile = "diagnostic_strict" if getattr(cfg, "strict", False) else "research_default"
        gate_results = assert_release_ready(rows_by_split, reports={"quality": quality}, leakage=leakage, profile=profile)
        
        # Save metrics for mandatory export
        metrics = {
            "counts": quality.export_counts,
            "quality": quality.__dict__ if hasattr(quality, "__dict__") else quality,
            "leakage": leakage,
            "gate_results": gate_results,
            "profile": profile
        }
        
        # Mandatory export of metrics_summary.json
        if not cfg.dry_run:
            with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
                # Need to handle non-serializable objects in quality report
                def d_ser(obj):
                    if hasattr(obj, "to_dict"): return obj.to_dict()
                    if hasattr(obj, "__dict__"): return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
                    return str(obj)
                json.dump(metrics, f, indent=2, default=d_ser)
        
        if cfg.dry_run:
            return {"counts": quality.export_counts, "quality": quality, "leakage": leakage, "dry_run": True}
            
        t2 = progress.add_task("[yellow]Exporting Artifacts...", total=4)
        counts = write_split_jsonl(output_dir, rows_by_split)
        progress.update(t2, advance=1)
        
        manifest = ArtifactManifest(
            version="dataset_builder_p0",
            dataset_inputs=[str(path) for path in cfg.input_paths],
            profile_summary=profile_summary or {},
            policies_used={
                "release_gate": "strict",
                "llm_provider": cfg.llm_provider,
                "llm_model": cfg.llm_model,
                "random_seed": cfg.random_seed,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "sample_size": cfg.sample_size,
                "chunk_size": cfg.chunk_size,
                "chunk_offset": cfg.chunk_offset,
            },
            split_summary=counts,
            release_status="passed",
        )
        write_manifest(output_dir / "manifest.json", manifest)
        progress.update(t2, advance=1)
        
        write_sidecar(output_dir / "quality_report.json", quality)
        progress.update(t2, advance=1)
        
        archive_path = write_artifact_zip(output_dir)
        progress.update(t2, advance=1)
        
    return {"counts": counts, "quality": quality, "leakage": leakage, "archive_path": archive_path}
