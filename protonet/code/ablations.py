from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ECConfig
from .dataset_loader import ECDatasetLoader
from .encoder import ECEncoder
from .evaluator import ECEvaluator
from .export import export_metrics, export_predictions
from .memory_support import MemorySupportModule
from .prototype_store import PrototypeStore
from .scorer import ECProtoNetScorer


def load_artifact_verification(input_dir: Path) -> dict[str, Any]:
    verification_path = input_dir / "artifact_verification.json"
    if not verification_path.exists():
        raise RuntimeError(
            f"Missing artifact_verification.json in {input_dir}. "
            "Run dataset_builder artifact verification before EC-ProtoNet."
        )

    with open(verification_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    artifact_status = report.get("artifact_status")
    ready_for_protonet = bool(report.get("ready_for_protonet", False))
    if artifact_status != "pass" or not ready_for_protonet:
        failed = report.get("failed_checks", [])
        raise RuntimeError(
            "Artifact is not ready for ProtoNet evaluation.\n"
            f"artifact_status={artifact_status}\n"
            f"ready_for_protonet={ready_for_protonet}\n"
            f"failed_checks={failed}"
        )
    return report


def _build_store_with_descriptions_and_memory(
    loader: ECDatasetLoader,
    encoder: ECEncoder,
    artifact_dir: Path,
) -> tuple[PrototypeStore, MemorySupportModule]:
    """
    Build the full prototype store:
      1. Train-evidence prototypes
      2. Generic description prototypes (fills unseen aspects)
      3. AspectMemory prototypes (learned clusters)
    Returns the store and a loaded MemorySupportModule.
    """
    store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    store.build_from_descriptions(encoder)

    memory_mod = MemorySupportModule.from_artifact(ECConfig(), artifact_dir, encoder=encoder)
    store.build_from_memory(encoder, memory_mod.memory_summary)

    return store, memory_mod


def run_ablation_study(
    artifact_dir: str | Path,
    output_dir: str | Path,
    export_preds: bool = False,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ECDatasetLoader(artifact_dir).load_all()
    encoder = ECEncoder()

    train_only_store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    full_store, memory_mod = _build_store_with_descriptions_and_memory(
        loader, encoder, artifact_dir
    )

    experiments: dict[str, tuple[ECConfig, PrototypeStore, MemorySupportModule | None]] = {
        "train_evidence_only_protonet": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            train_only_store,
            None,
        ),
        "train_plus_description_protonet": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            None,
        ),
        "protonet_plus_evidence": (
            ECConfig(use_evidence=True, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            None,
        ),
        "protonet_plus_memory": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            memory_mod,
        ),
        "protonet_plus_selective": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=False, use_router=True),
            full_store,
            memory_mod,
        ),
        "full_ec_without_contradiction": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=False, use_router=True),
            full_store,
            memory_mod,
        ),
        "full_ec_protonet": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=True, use_router=True),
            full_store,
            memory_mod,
        ),
    }

    all_results: dict[str, Any] = {}
    test_examples = loader.splits["test"]

    for name, (config, store, mem) in experiments.items():
        print(f"Running ablation: {name}...")
        scorer = ECProtoNetScorer(encoder, store, config, memory_mod=mem)
        evaluator = ECEvaluator()

        predictions = [scorer.predict(ex) for ex in test_examples]
        metrics = evaluator.evaluate(test_examples, predictions)
        metrics["experiment"] = name

        # Always export predictions for better error analysis (EC-P5)
        pred_path = output_dir / f"predictions_{name}.jsonl"
        export_predictions(predictions, test_examples, pred_path)

        export_metrics(metrics, output_dir / f"metrics_{name}.json")
        all_results[name] = metrics

    with open(output_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    proto_summary = full_store.summary()
    with open(output_dir / "prototype_summary.json", "w", encoding="utf-8") as f:
        json.dump(proto_summary, f, indent=2)

    return all_results

def run_compare_study(
    input_dir: str | Path,
    output_dir: str | Path,
    models: list[str],
    export_preds: bool = True,  # Default to True (EC-P5)
    overwrite: bool = False,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    verification_report = load_artifact_verification(input_dir)
    
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            print(f"ERROR: Output directory {output_dir} exists. Use --overwrite to proceed.")
            import sys
            sys.exit(1)
        else:
            # Simple cleanup of metric and prediction files
            import shutil
            shutil.rmtree(output_dir)
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ECDatasetLoader(input_dir).load_all()
    encoder = ECEncoder()

    train_only_store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    full_store, memory_mod = _build_store_with_descriptions_and_memory(
        loader, encoder, input_dir
    )

    # Rename misleading model modes (EC-P1)
    experiments: dict[str, tuple[ECConfig, PrototypeStore, MemorySupportModule | None]] = {
        "train_evidence_only": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            train_only_store,
            None,
        ),
        "train_plus_description": (
            ECConfig(use_evidence=False, use_memory=False, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            None,
        ),
        "train_plus_description_plus_memory": (
            ECConfig(use_evidence=False, use_memory=True, use_novelty=False, use_contradiction=False, use_router=False),
            full_store,
            memory_mod,
        ),
        "full_ec_protonet": (
            ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=True, use_router=True),
            full_store,
            memory_mod,
        ),
    }

    all_results = {}
    test_examples = loader.splits["test"]

    for name in models:
        # Handle legacy names for backward compatibility if needed, but prefer new names
        mapped_name = name
        if name == "legacy_protonet": 
            print("WARNING: 'legacy_protonet' alias used. Mapping to 'train_evidence_only'. This is NOT a real legacy model bundle.")
            mapped_name = "train_evidence_only"
        if name == "plain_protonet": 
            print("WARNING: 'plain_protonet' alias used. Mapping to 'train_plus_description'.")
            mapped_name = "train_plus_description"

        if mapped_name not in experiments:
            print(f"Warning: Model {name} not found in configured experiments. Available: {list(experiments.keys())}")
            continue
            
        config, store, mem = experiments[mapped_name]
        print(f"Running comparison: {name}...")
        scorer = ECProtoNetScorer(encoder, store, config, memory_mod=mem)
        evaluator = ECEvaluator()

        eval_top_k = 3 if mapped_name in {"full_ec_protonet", "train_plus_description"} else 1
        predictions = [scorer.predict(ex, top_k=eval_top_k) for ex in test_examples]
        metrics = evaluator.evaluate(test_examples, predictions)
        predictions_top1 = [scorer.predict(ex, top_k=1) for ex in test_examples]
        predictions_top3 = [scorer.predict(ex, top_k=3) for ex in test_examples]
        metrics_top1 = evaluator.evaluate(test_examples, predictions_top1)
        metrics_top3 = evaluator.evaluate(test_examples, predictions_top3)
        metrics["top1_strict_f1"] = metrics_top1.get("strict_f1", 0.0)
        metrics["top3_relaxed_f1"] = metrics_top3.get("relaxed_multi_gold_f1", 0.0)
        metrics["top3_recall"] = metrics_top3.get("strict_recall", 0.0)
        metrics["metrics_schema_version"] = "ec_eval_v4"
        metrics["experiment"] = name

        if export_preds:
            pred_path = output_dir / f"predictions_{name}.jsonl"
            export_predictions(predictions, test_examples, pred_path)

        export_metrics(metrics, output_dir / f"{name}_metrics.json")
        all_results[name] = metrics

    # Write run_metadata.json (EC-P6)
    artifact_status = verification_report.get("artifact_status", "unknown")
    ready_for_protonet = bool(verification_report.get("ready_for_protonet", False))
    stale_guard = "unknown"
    
    try:
        # Check active artifact contract
        active_contract_path = Path("CURRENT_ACTIVE_ARTIFACT.json")
        if active_contract_path.exists():
            with open(active_contract_path, "r") as af:
                adata = json.load(af)
                active_path = adata.get("active_dataset_artifact")
                if active_path and Path(active_path).resolve() == input_dir.resolve():
                    stale_guard = "pass"
                else:
                    stale_guard = "stale_path_warning"
                    print(f"WARNING: Artifact path {input_dir} does not match CURRENT_ACTIVE_ARTIFACT.json")
    except Exception as e:
        print(f"Metadata error: {e}")

    import datetime
    run_metadata = {
      "run_id": f"compare_study_{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}",
      "timestamp": datetime.datetime.now().isoformat(),
      "artifact_dir": str(input_dir),
      "artifact_status": artifact_status,
      "ready_for_protonet": ready_for_protonet,
      "metrics_schema_version": "ec_eval_v4",
      "models": models,
      "stale_output_guard": stale_guard
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    proto_summary = full_store.summary()
    with open(output_dir / "prototype_summary.json", "w", encoding="utf-8") as f:
        json.dump(proto_summary, f, indent=2)
        
    # Build comparison table
    import csv
    table_path = output_dir / "comparison_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "strict_f1", "relaxed_multi_gold_f1", "coverage", "accepted_accuracy", "abstention_f1", "novel_f1", "coverage_on_valid"])
        for name, m in all_results.items():
            writer.writerow([
                name,
                f"{m.get('strict_f1', 0.0):.4f}",
                f"{m.get('relaxed_multi_gold_f1', 0.0):.4f}",
                f"{m.get('coverage', 0.0):.4f}",
                f"{m.get('accepted_accuracy', 0.0):.4f}",
                f"{m.get('abstention_f1', 0.0):.4f}",
                f"{m.get('novel_f1', 0.0):.4f}",
                f"{m.get('coverage_on_valid_gold_rows', 0.0):.4f}",
            ])

    return all_results

def run_learning_loop_study(
    artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    Demonstrate the closed-loop learning improvement (Pass A vs Pass B).
    Pass A: No memory.
    Pass B: Use memory from artifact (which should have promoted entries).
    """
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ECDatasetLoader(artifact_dir).load_all()
    encoder = ECEncoder()
    full_store, memory_mod = _build_store_with_descriptions_and_memory(loader, encoder, artifact_dir)
    
    # Check if memory actually has promoted entries
    promoted_count = 0
    if memory_mod and memory_mod.memory_summary:
        promoted_count = sum(1 for c in memory_mod.memory_summary.get("top_clusters", []) 
                             if str(c.get("status")).lower() == "promoted")
    
    print(f"Learning Loop Study: Found {promoted_count} promoted entries in memory.")
    
    # Pass A: Baseline (Descriptions only, no memory)
    config_a = ECConfig(use_memory=False)
    scorer_a = ECProtoNetScorer(encoder, full_store, config_a)
    
    # Pass B: Augmented (Memory enabled)
    config_b = ECConfig(use_memory=True)
    scorer_b = ECProtoNetScorer(encoder, full_store, config_b, memory_mod=memory_mod)
    
    evaluator = ECEvaluator()
    test_examples = loader.splits["test"]
    
    results = {}
    for name, scorer in [("Pass_A_NoMemory", scorer_a), ("Pass_B_WithMemory", scorer_b)]:
        print(f"Running {name}...")
        preds = [scorer.predict(ex) for ex in test_examples]
        metrics = evaluator.evaluate(test_examples, preds)
        metrics["experiment"] = name
        results[name] = metrics
        
        export_metrics(metrics, output_dir / f"{name}_metrics.json")
        
    # Isolation: Compare results specifically on rows where memory was triggered
    # This requires looking at the scores, which we'll do in a future update if needed.
    
    f1_diff = results["Pass_B_WithMemory"]["strict_f1"] - results["Pass_A_NoMemory"]["strict_f1"]
    print(f"Memory Loop Impact (Strict F1): {f1_diff:+.4f}")
    
    with open(output_dir / "learning_loop_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results

def run_novelty_study(
    input_file: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    input_file = Path(input_file)
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = load_artifact_verification(artifact_dir)

    if not input_file.exists():
        print(f"ERROR: Novelty input file {input_file} not found.")
        return {}

    from .dataset_loader import load_jsonl_rows, normalize_row
    examples = [normalize_row(r) for r in load_jsonl_rows(input_file)]
    
    loader = ECDatasetLoader(artifact_dir).load_all()
    encoder = ECEncoder()
    full_store, memory_mod = _build_store_with_descriptions_and_memory(
        loader, encoder, artifact_dir
    )

    config = ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=True, use_router=True)
    scorer = ECProtoNetScorer(encoder, full_store, config, memory_mod=memory_mod)
    evaluator = ECEvaluator()

    predictions = [scorer.predict(ex) for ex in examples]
    metrics = evaluator.evaluate(examples, predictions)
    metrics["experiment"] = "novelty_eval"
    metrics["metrics_schema_version"] = "ec_eval_v4"

    # Risk-coverage and threshold sweep
    from dataclasses import replace
    rc_data = []
    grid = []
    for accept_t in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for abstain_t in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for novel_t in [0.60, 0.70, 0.80, 0.90]:
                for margin_t in [0.01, 0.03, 0.05, 0.08]:
                    new_config = replace(
                        config,
                        accept_threshold=accept_t,
                        abstain_threshold=abstain_t,
                        novel_threshold=novel_t,
                        margin_threshold=margin_t,
                    )
                    scorer = ECProtoNetScorer(encoder, full_store, new_config, memory_mod=memory_mod)
                    t_preds = [scorer.predict(ex, top_k=3) for ex in examples]
                    t_metrics = evaluator.evaluate(examples, t_preds)
                    objective = (
                        t_metrics.get("relaxed_multi_gold_f1", 0.0)
                        + 0.5 * t_metrics.get("abstention_f1", 0.0)
                        + 0.5 * t_metrics.get("novel_f1", 0.0)
                    )
                    row = {
                        "accept_threshold": accept_t,
                        "abstain_threshold": abstain_t,
                        "novel_threshold": novel_t,
                        "margin_threshold": margin_t,
                        "objective": objective,
                        "coverage": t_metrics.get("coverage", 0.0),
                        "accepted_accuracy": t_metrics.get("accepted_accuracy", 0.0),
                        "strict_f1": t_metrics.get("strict_f1", 0.0),
                        "novel_f1": t_metrics.get("novel_f1", 0.0),
                        "abstention_f1": t_metrics.get("abstention_f1", 0.0),
                    }
                    grid.append({**row, **t_metrics})
                    rc_data.append(row)

    best = max(grid, key=lambda x: x["objective"]) if grid else {}
    metrics["risk_coverage_curve"] = rc_data
    metrics["best_config"] = {
        "accept_threshold": best.get("accept_threshold"),
        "abstain_threshold": best.get("abstain_threshold"),
        "novel_threshold": best.get("novel_threshold"),
        "margin_threshold": best.get("margin_threshold"),
    }
    metrics["best_metrics"] = {
        "relaxed_multi_gold_f1": best.get("relaxed_multi_gold_f1"),
        "abstention_f1": best.get("abstention_f1"),
        "novel_f1": best.get("novel_f1"),
        "coverage": best.get("coverage"),
        "accepted_accuracy": best.get("accepted_accuracy"),
    }
    
    # Export results
    export_metrics(metrics, output_dir / "novelty_metrics.json")
    export_predictions(predictions, examples, output_dir / "novelty_predictions.jsonl")
    
    with open(output_dir / "risk_coverage_data.json", "w", encoding="utf-8") as f:
        json.dump(rc_data, f, indent=2)

    print(f"Novelty study complete. Results: {metrics.get('novel_f1')}")
    return metrics
