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
from .io_utils import source_code_hash, sha256_tree, write_json, write_jsonl
from .known_classifier import build_known_classifier
from .memory import build_memory_index
from .prototype_store import augment_with_memory_prototypes, build_prototype_store
from .provenance import sha256_source_tree
from .router import SelectiveRouterV2
from .schema import PredictionRecord, ReviewExample, to_runtime_example
from .scorer import ScoringContext, compute_energy_stats, score_examples


def protocol_train_rows(bundle: DatasetBundle, protocol: str) -> list[ReviewExample]:
    if protocol == "grouped":
        return bundle.splits["train"]
    if protocol == "domain_holdout":
        return bundle.domain_holdout["train"]
    raise ValueError(f"Unknown protocol: {protocol}")


def protocol_eval_rows(bundle: DatasetBundle, protocol: str, split: str) -> list[ReviewExample]:
    if protocol == "grouped":
        return bundle.splits[split]
    if protocol == "domain_holdout":
        return bundle.domain_holdout[split]
    raise ValueError(f"Unknown protocol: {protocol}")


def build_context(
    bundle: DatasetBundle, config: ECProtoNetV2Config, *, train_rows: list[ReviewExample] | None = None
) -> ScoringContext:
    train_rows = train_rows if train_rows is not None else bundle.train
    encoder = build_encoder(config.encoder, config.model_name, config.normalize_embeddings, config.hashing_dim, config.batch_size)
    from .evidence_distilled_profiles import distill_aspect_profiles
    from .aspect_graph import build_aspect_graph
    from .import schema

    distilled_profiles = distill_aspect_profiles(train_rows, config)
    aspect_graph = build_aspect_graph(train_rows, encoder, config) if getattr(config, "use_aspect_graph", True) else None
    prototypes = build_prototype_store(
        train_rows,
        encoder,
        config,
        label_equivalence=bundle.label_equivalence,
        distilled_profiles=distilled_profiles,
        aspect_graph=aspect_graph,
    )
    schema.ACTIVE_ASPECT_GRAPH = aspect_graph
    memory_items = []
    seen_clusters = set()
    memory_mode = str(getattr(config, "memory_source_mode", "promoted") or "promoted").lower()
    if memory_mode == "review_queue":
        memory_sources = [bundle.aspect_memory_review_queue]
    elif memory_mode == "candidates":
        memory_sources = [bundle.aspect_memory_candidates]
    elif memory_mode == "oracle_all":
        # Research-only oracle ablation: intentionally mixes all memory pools.
        memory_sources = [bundle.aspect_memory_promoted, bundle.aspect_memory_review_queue, bundle.aspect_memory_candidates]
    else:
        memory_sources = [bundle.aspect_memory_promoted]

    for source in memory_sources:
        for item in source:
            cid = item.get("cluster_id")
            if cid and cid in seen_clusters:
                continue
            if cid:
                seen_clusters.add(cid)
            memory_items.append(item)
    memory = build_memory_index(memory_items, bundle.normalizer, encoder, config) if config.use_memory else None
    if (
        config.use_memory
        and bool(getattr(config, "use_memory_prototypes", False))
        and memory is not None
    ):
        prototypes = augment_with_memory_prototypes(prototypes, memory, config)
    known_classifier = (
        build_known_classifier(
            train_rows,
            bundle.normalizer,
            encoder,
            max_labels=int(getattr(config, "known_classifier_max_labels", 50)),
        )
        if getattr(config, "use_known_classifier", False)
        else None
    )
    
    # Calculate priors (Phase 7)
    counts = {}
    total = 0
    for ex in train_rows:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                counts[g.aspect] = counts.get(g.aspect, 0) + 1
                total += 1
    priors = {a: c / max(1, total) for a, c in counts.items()}
    new_config = replace(
        config,
        aspect_priors=priors,
        open_world_alias_map=bundle.label_equivalence or {},
    )

    return ScoringContext(
        encoder=encoder,
        prototypes=prototypes,
        memory=memory,
        config=new_config,
        known_classifier=known_classifier,
        label_aliases=bundle.label_equivalence,
        distilled_profiles=distilled_profiles,
    )


def predict_records(examples: list[ReviewExample], context: ScoringContext) -> list[PredictionRecord]:
    runtime_examples = [to_runtime_example(ex) for ex in examples]
    raw_scores = score_examples(runtime_examples, context, top_k=context.config.top_k)
    router = SelectiveRouterV2(context.config, sibling_confusion=getattr(context, "sibling_confusion", None))
    records: list[PredictionRecord] = []
    for ex, runtime_ex in zip(examples, runtime_examples):
        cands = raw_scores.get(ex.row_id, [])
        if getattr(context.config, "use_candidate_reranker", False) and context.candidate_reranker is not None:
            top1_aspect = cands[0].aspect if cands else ""
            other_aspects = {c.aspect for c in cands}
            cands = [
                replace(
                    c, 
                    reranker_score=context.candidate_reranker.score(
                        c, 
                        source_type=getattr(runtime_ex, "source_type", "implicit"),
                        top1_aspect=top1_aspect,
                        other_aspects=other_aspects,
                        sibling_confusion_matrix=getattr(context, "sibling_confusion", None),
                        other_candidates=cands,
                        label_descriptions=context.prototypes.label_descriptions,
                        review_text=runtime_ex.text,
                        config=context.config,
                    )
                ) 
                for c in cands
            ]
        routed = router.route_candidates(runtime_ex, cands)
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
    protocol: str = "grouped",
    allow_failed_artifact: bool = False,
    allow_stale_source_artifact: bool = False,
) -> dict[str, Any]:
    if allow_failed_artifact:
        config = replace(config, require_artifact_pass=False, require_active_contract=False)
    guard = assert_verified_artifact(
        artifact_dir,
        config,
        allow_stale_source_artifact=allow_stale_source_artifact,
    )
    bundle = load_dataset_bundle(artifact_dir)
    train_rows = protocol_train_rows(bundle, protocol)
    context = build_context(bundle, config, train_rows=train_rows)
    calibration_examples = protocol_eval_rows(bundle, protocol, "val")
    calibration_runtime = [to_runtime_example(ex) for ex in calibration_examples]
    context.energy_stats = compute_energy_stats(calibration_runtime, context)

    if getattr(config, "use_candidate_reranker", False):
        from .candidate_reranker import train_candidate_reranker, build_sibling_confusion_matrix
        val_scores = score_examples(calibration_runtime, context, top_k=config.top_k)
        sibling_confusion = build_sibling_confusion_matrix(val_scores, calibration_examples)
        context.sibling_confusion = sibling_confusion
        
        # Rebuild aspect graph incorporating validation confusion and update schema
        from .aspect_graph import build_aspect_graph
        from .import schema
        if getattr(config, "use_aspect_graph", True):
            aspect_graph = build_aspect_graph(train_rows, context.encoder, config, validation_confusion=sibling_confusion)
        else:
            aspect_graph = None
        schema.ACTIVE_ASPECT_GRAPH = aspect_graph
        
        # If smoothed prototypes are enabled, rebuild the prototype store with the updated aspect graph
        if getattr(config, "use_graph_smoothed_prototypes", False):
            context.prototypes = build_prototype_store(
                train_rows,
                context.encoder,
                config,
                label_equivalence=bundle.label_equivalence,
                distilled_profiles=context.distilled_profiles,
                aspect_graph=aspect_graph,
            )
        
        rows_data = []
        labels_data = []
        for val_ex, val_run_ex in zip(calibration_examples, calibration_runtime):
            candidates = val_scores.get(val_ex.row_id, [])
            top1_aspect = candidates[0].aspect if candidates else ""
            other_aspects = {c.aspect for c in candidates}
            gold_set = set(val_ex.gold_labels)
            for c in candidates:
                rows_data.append((c, getattr(val_run_ex, "source_type", "implicit"), top1_aspect, other_aspects, candidates, val_run_ex.text))
                labels_data.append(1 if c.aspect in gold_set else 0)
        
        if len(labels_data) > 0 and len(set(labels_data)) > 1:
            reranker = train_candidate_reranker(
                rows_data, 
                labels_data, 
                sibling_confusion_matrix=sibling_confusion,
                label_descriptions=context.prototypes.label_descriptions,
                config=config,
            )
            context.candidate_reranker = reranker
        else:
            context.candidate_reranker = None

    examples = protocol_eval_rows(bundle, protocol, split)
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
            "protocol": protocol,
            "split": split,
            "config": config.to_dict(),
            "dataset_summary": bundle.summary(),
            "prototype_count": len(context.prototypes.aspects),
            "known_train_label_count": len(context.prototypes.aspects),
            "prediction_count": len(records),
            "unseen_test_rows": sum(1 for r in records if r.gold_unseen_labels),
            "emerging_test_rows": sum(1 for r in records if r.gold_emerging_labels),
            "memory_entry_count": 0 if context.memory is None else len(context.memory.entries),
            "code_hash": source_code_hash(Path(__file__).resolve().parent),
            "source_code_hash": sha256_source_tree(Path(__file__).resolve().parent),
            "config_hash": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
        },
    )
    if config.export_predictions:
        write_jsonl(output_dir / "predictions.jsonl", [r.to_dict() for r in records])

    open_world_rows = [r.to_dict() for r in records if r.has_open_world]
    if open_world_rows:
        write_jsonl(output_dir / "open_world_candidates.jsonl", open_world_rows)

    return {"metrics": metrics, "dataset_summary": bundle.summary(), "guard": guard.to_dict()}
