from __future__ import annotations
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("V7_Runner")

# Ensure code is in path
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# V7 Imports
try:
    from row_contracts import RawLoaded, Prepared, Grounded, ImplicitScored, SplitAssigned
    from ingestion.io_utils import load_inputs
    from extraction.explicit_features import build_explicit_row
    from extraction.implicit_pipeline import build_implicit_row
    from curation.scoring import apply_quality_scoring
    from curation.bucketing import assign_bucket
    from curation.deduplication import apply_semantic_dedup
    from splitting.splitter import v7_split_pipeline
    from benchmark.exporters import v7_export_pipeline, write_benchmark_outputs
    from contracts import BuilderConfig
except ImportError as e:
    logger.error(f"Failed to import V7 modules: {e}")
    raise

from extraction.explicit_features import fit_explicit_artifacts, ExplicitArtifacts
import asyncio


def run_pipeline_sync(cfg: Any, *, pipeline: Any) -> Any:
    result = pipeline(cfg)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


async def run_v7_pipeline_from_cfg(cfg: BuilderConfig) -> dict[str, Any]:
    """Orchestrate the V7 dataset builder pipeline."""
    
    # Ensure output directories exist
    cfg.ensure_dirs(reset_output=cfg.reset_output)
    
    # 1. Ingestion
    logger.info(f"Phase 1: Ingestion from {cfg.input_dir}")
    if not cfg.input_dir.exists():
        logger.warning("Input directory does not exist.")
        return {"error": "no_inputs"}
        
    df = load_inputs(cfg.input_dir)
    if df.empty:
        logger.warning("No input data loaded.")
        return {"error": "no_inputs"}
        
    raw_dicts = df.to_dict("records")
    if cfg.sample_size:
        raw_dicts = raw_dicts[:cfg.sample_size]
        
    raw_rows = []
    for i, d in enumerate(raw_dicts):
        # Handle potential NaN values from pandas conversion
        source_id = str(d.get("id") or d.get("instance_id") or f"src_{i}")
        if source_id == "nan": source_id = f"src_{i}"
        
        text = str(d.get("text") or d.get("review_text") or "")
        
        raw_rows.append(RawLoaded(
            source_id=source_id,
            review_text=text,
            metadata=d
        ))
    logger.info(f"Initialized {len(raw_rows)} RawLoaded rows.")
    
    # 2. Preparation (Basic cleaning and ID assignment)
    prepared_rows = []
    for i, r in enumerate(raw_rows):
        domain = str(r.metadata.get("domain") or "unknown")
        group_id = str(r.metadata.get("group_id") or r.metadata.get("id") or f"G_{i}")
        if group_id == "nan": group_id = f"G_{i}"
        
        prepared_rows.append(Prepared(
            row_id=f"V7_{i}",
            review_text=r.review_text,
            domain=domain,
            group_id=group_id
        ))
    logger.info(f"Prepared {len(prepared_rows)} rows for extraction.")
    
    # 3. Extraction Artifacts
    logger.info("Initializing extraction artifacts...")
    # Map back to df for fitting if needed, or use full df
    # For now we use the full df we loaded
    exp_artifacts = fit_explicit_artifacts(df, [], []) # Pass empty numeric/categorical for now
    
    # 4. Extraction Phase
    logger.info("Phase 2: Extraction (Explicit + Implicit)")
    extracted_rows = []
    
    # For performance in a real run, we'd use gather() but let's keep it simple for now
    for i, row in enumerate(prepared_rows):
        # 4a. Try Explicit
        grounded = build_explicit_row(row, artifacts=exp_artifacts)
        if grounded:
            extracted_rows.append(grounded)
            continue
            
        # 4b. Fallback to Implicit Discovery
        # We need candidate aspects - we'll use a default set or from registry
        # For this iteration, we use the ones from the implicit pipeline's defaults
        from extraction.implicit_pipeline import VALID_LATENT_ASPECTS
        candidate_aspects = list(VALID_LATENT_ASPECTS)
        
        try:
            imp_row = await build_implicit_row(
                row,
                candidate_aspects=candidate_aspects,
                confidence_threshold=cfg.confidence_threshold,
                row_index=i,
                domain=row.domain,
                implicit_mode=cfg.implicit_mode,
                use_coref=cfg.use_coref,
                enable_llm_fallback=cfg.enable_llm_fallback,
                llm_fallback_threshold=cfg.llm_fallback_threshold,
                llm_provider=cfg.llm_provider,
                llm_model_name=cfg.llm_model_name
            )
            extracted_rows.append(imp_row)
        except Exception as e:
            logger.error(f"Implicit extraction failed for row {row.row_id}: {e}")
            # We don't want the whole pipeline to crash if one row fails
            continue
            
    logger.info(f"Extraction complete. {len(extracted_rows)} processed.")
    
    # 5. Curation
    logger.info("Phase 3: Curation")
    scored_rows = [apply_quality_scoring(r) for r in extracted_rows]
    bucketed_rows = [assign_bucket(r) for r in scored_rows]
    deduped_rows = apply_semantic_dedup(bucketed_rows)
    
    # 6. Splitting
    logger.info(f"Phase 4: Splitting ({cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio})")
    split_rows = v7_split_pipeline(
        deduped_rows,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        random_seed=cfg.random_seed
    )
    
    # 7. Export
    logger.info("Phase 5: Exporting artifacts")
    benchmark_instances, train_examples = v7_export_pipeline(split_rows)
    
    # Prepare rows by split for benchmark output
    rows_by_split = {"train": [], "val": [], "test": []}
    for bm in benchmark_instances:
        rows_by_split[bm.metadata.get("split", "test")].append(bm.model_dump())
        
    write_benchmark_outputs(
        cfg.benchmark_dir,
        rows_by_split,
        metadata={
            "pipeline": "V7_Modular",
            "rows": len(benchmark_instances),
            "train_examples": len(train_examples)
        }
    )
    
    # Save train examples
    train_path = cfg.output_dir / "trainset.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex.model_dump()) + "\n")
            
    logger.info(f"V7 Pipeline Complete. Benchmark: {len(benchmark_instances)}, Train: {len(train_examples)}")
    return {
        "benchmark_count": len(benchmark_instances),
        "train_count": len(train_examples)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--sample_size", type=int)
    args = parser.parse_args()
    
    cfg = BuilderConfig()
    if args.input_dir: cfg.input_dir = Path(args.input_dir)
    if args.output_dir: cfg.output_dir = Path(args.output_dir)
    if args.sample_size: cfg.sample_size = args.sample_size
    
    asyncio.run(run_v7_pipeline_from_cfg(cfg))
