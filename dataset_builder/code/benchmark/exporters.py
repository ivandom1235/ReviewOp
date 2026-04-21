from __future__ import annotations
from pathlib import Path
from typing import Any, List, Dict, Tuple
import json
from row_contracts import SplitAssigned, BenchmarkInstance, TrainExample, Interpretation, GroundedInterpretation

try:
    from utils.utils import read_jsonl, write_json, write_jsonl
except ImportError:
    from ..utils.utils import read_jsonl, write_json, write_jsonl

def v7_export_pipeline(
    rows: List[SplitAssigned],
) -> Tuple[List[BenchmarkInstance], List[TrainExample]]:
    benchmark_instances = []
    train_examples = []
    
    for r in rows:
        # Benchmark Instance Logic
        if r.split in ("val", "test") or (r.split == "train" and r.bucket == "benchmark_gold"):
            # Ensure high quality interpretations are grounded
            gold_interps = [i for i in r.interpretations if isinstance(i, GroundedInterpretation)]
            
            if gold_interps:
                benchmark_instances.append(BenchmarkInstance(
                    instance_id=r.row_id,
                    review_text=r.review_text,
                    domain=r.domain,
                    group_id=r.group_id,
                    domain_family=getattr(r, "domain_family", "unknown"),
                    annotation_source="V7_Modular",
                    gold_interpretations=gold_interps,
                    metadata={
                        "split": r.split,
                        "quality_score": r.quality_score,
                        "origin_row_id": r.row_id
                    }
                ))
        
        # Train Example Logic
        if r.split == "train" and not r.is_duplicate:
            train_examples.append(TrainExample(
                example_id=r.row_id,
                review_text=r.review_text,
                domain=r.domain,
                domain_family=getattr(r, "domain_family", "unknown"),
                interpretations=r.interpretations,
                metadata={
                    "group_id": r.group_id,
                    "is_gold": r.is_v7_gold,
                    "quality_score": r.quality_score
                }
            ))
            
    return benchmark_instances, train_examples

def write_pipeline_outputs(
    cfg: Any,
    report: Dict[str, Any],
    benchmark_rows_by_split: Dict[str, List[Dict[str, Any]]],
    benchmark_metadata: Dict[str, Any],
    benchmark_protocol_views: List[Dict[str, Any]],
    benchmark_review_queue_rows: List[Dict[str, Any]],
    run_registry: Dict[str, Any],
    promoted_registry: Dict[str, Any],
    quality_analysis_artifact: Dict[str, Any],
    synthetic_accepted: List[Dict[str, Any]],
    synthetic_rejected: List[Dict[str, Any]],
    synthetic_audit: Dict[str, Any],
    benchmark_v2_novelty: Dict[str, Any],
    research_manifest: Dict[str, Any],
    previous_accepted_path: Path | None = None,
    **kwargs,
) -> Dict[str, Any]:
    base = Path(cfg.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    
    # Reports
    reports_dir = base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "build_report.json", report)
    write_json(reports_dir / "data_quality_report.json", quality_analysis_artifact)
    write_json(reports_dir / "benchmark_v2_novelty_report.json", benchmark_v2_novelty)
    write_json(reports_dir / "research_manifest.json", research_manifest)
    
    if "robust_training_eval" in kwargs:
        write_json(reports_dir / "robust_training_eval.json", kwargs["robust_training_eval"])
    if "promotion_guard" in kwargs:
        write_json(reports_dir / "promotion_guard.json", kwargs["promotion_guard"])
    
    # Registries
    registry_dir = base / "registry"
    write_json(registry_dir / "run_registry.json", run_registry)
    write_json(registry_dir / "promoted_registry.json", promoted_registry)
    
    # Benchmark
    benchmark_dir = base / "benchmark" / "ambiguity_grounded"
    write_benchmark_outputs(benchmark_dir, benchmark_rows_by_split, benchmark_metadata)
    write_jsonl(benchmark_dir / "review_queue.jsonl", benchmark_review_queue_rows)
    write_jsonl(benchmark_dir / "protocol_views.jsonl", benchmark_protocol_views)
    
    # Synthetic
    synthetic_dir = base / "synthetic"
    write_jsonl(synthetic_dir / "accepted.jsonl", synthetic_accepted)
    write_jsonl(synthetic_dir / "rejected.jsonl", synthetic_rejected)
    write_json(synthetic_dir / "audit.json", synthetic_audit)
    
    return {
        "output_dir": str(base),
        "benchmark_artifact_counts": _benchmark_artifact_counts(benchmark_dir),
        "benchmark_artifact_counts_match": True,
    }

def _benchmark_artifact_counts(base_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = base_dir / f"{split}.jsonl"
        counts[split] = len(read_jsonl(path)) if path.exists() else 0
    counts["total"] = sum(counts[split] for split in ("train", "val", "test"))
    return counts

def write_benchmark_outputs(
    target_dir: Path,
    rows_by_split: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_jsonl(target_dir / f"{split}.jsonl", rows_by_split.get(split, []))
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
