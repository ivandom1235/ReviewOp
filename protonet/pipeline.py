from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import hashlib
import json

from .artifact_guard import ArtifactGuardReport, assert_verified_artifact
from .config import ECProtoNetV2Config
from .dataset import DatasetBundle, load_dataset_bundle
from .encoder import build_encoder
from .evaluator import evaluate_predictions
from .io_utils import sha256_tree, write_json, write_jsonl
from .memory import build_memory_index
from .prototype_store import build_prototype_store
from .router import SelectiveRouterV2
from .schema import PredictionRecord, ReviewExample
from .scorer import ScoringContext, score_examples


def build_context(bundle: DatasetBundle, config: ECProtoNetV2Config) -> ScoringContext:
    encoder = build_encoder(config.encoder, config.model_name, config.normalize_embeddings, config.hashing_dim, config.batch_size)
    prototypes = build_prototype_store(bundle.train, encoder, config, label_equivalence=bundle.label_equivalence)
    memory_items = []
    seen_clusters = set()
    for source in [
        bundle.aspect_memory_promoted,
        bundle.aspect_memory_review_queue,
        bundle.aspect_memory_candidates,
        bundle.aspect_memory_summary.get("top_clusters", []),
    ]:
        for item in source:
            cid = item.get("cluster_id")
            if cid and cid in seen_clusters:
                continue
            if cid:
                seen_clusters.add(cid)
            memory_items.append(item)
    memory = build_memory_index(memory_items, bundle.normalizer, encoder, config) if config.use_memory else None
    
    # Calculate priors (Phase 7)
    counts = {}
    total = 0
    for ex in bundle.train:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                counts[g.aspect] = counts.get(g.aspect, 0) + 1
                total += 1
    priors = {a: c / max(1, total) for a, c in counts.items()}
    new_config = replace(config, aspect_priors=priors)

    return ScoringContext(encoder=encoder, prototypes=prototypes, memory=memory, config=new_config)



def predict_records(examples: list[ReviewExample], context: ScoringContext) -> list[PredictionRecord]:
    raw_scores = score_examples(examples, context, top_k=context.config.top_k)
    router = SelectiveRouterV2(context.config)
    records: list[PredictionRecord] = []
    for ex in examples:
        routed = router.route_candidates(ex, raw_scores.get(ex.row_id, []))
        records.append(
            PredictionRecord(
                row_id=ex.row_id,
                review_id=ex.review_id,
                split=ex.split,
                domain=ex.domain,
                text=ex.text,
                gold_labels=sorted(ex.gold_labels),
                gold_novel_labels=sorted(ex.gold_novel_labels),
                gold_boundary_labels=sorted(ex.gold_boundary_labels),
                gold_emerging_labels=list(ex.gold_emerging_labels),
                gold_unseen_labels=ex.gold_unseen_labels,
                abstain_acceptable=ex.abstain_acceptable,
                candidates=routed,
            )
        )
    return records


def prototype_inventory(context: ScoringContext) -> dict:
    return {
        "prototype_count": len(context.prototypes.aspects),
        "aspects": context.prototypes.aspects,
        "support_counts": context.prototypes.support_counts,
        "prototype_source": context.prototypes.prototype_source,
    }


def run_compare(
    artifact_dir: str | Path,
    output_dir: str | Path,
    config: ECProtoNetV2Config,
    split: str = "test",
    allow_failed_artifact: bool = False,
) -> dict[str, Any]:
    if allow_failed_artifact:
        config = replace(config, require_artifact_pass=False, require_active_contract=False)
    guard = assert_verified_artifact(artifact_dir, config)
    bundle = load_dataset_bundle(artifact_dir)
    context = build_context(bundle, config)
    examples = getattr(bundle, split)
    records = predict_records(examples, context)
    metrics = evaluate_predictions(records, bundle.normalizer, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "dataset_summary.json", bundle.summary())
    write_json(output_dir / "artifact_guard_report.json", guard.to_dict())

    if config.export_prototype_inventory:
        write_json(output_dir / "prototype_inventory.json", prototype_inventory(context))

    write_json(
        output_dir / "run_metadata.json",
        {
            "metrics_schema_version": config.metrics_schema_version,
            "artifact_dir": str(Path(artifact_dir).resolve()),
            "artifact_status": guard.artifact_status,
            "ready_for_protonet": guard.ready_for_protonet,
            "stale_output_guard": guard.stale_output_guard,
            "artifact_sha256": guard.artifact_sha256,
            "config": config.to_dict(),
            "dataset_summary": bundle.summary(),
            "prototype_count": len(context.prototypes.aspects),
            "known_train_label_count": len(context.prototypes.aspects),
            "prediction_count": len(records),
            "unseen_test_rows": sum(1 for r in records if r.gold_unseen_labels),
            "emerging_test_rows": sum(1 for r in records if r.gold_emerging_labels),
            "memory_entry_count": 0 if context.memory is None else len(context.memory.entries),
            "code_hash": sha256_tree(Path(__file__).resolve().parent),
            "config_hash": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
        },
    )
    if config.export_predictions:
        write_jsonl(output_dir / "predictions.jsonl", [r.to_dict() for r in records])

    open_world_rows = [r.to_dict() for r in records if r.has_open_world]
    if open_world_rows:
        write_jsonl(output_dir / "open_world_candidates.jsonl", open_world_rows)

    return {"metrics": metrics, "dataset_summary": bundle.summary(), "guard": guard.to_dict()}
