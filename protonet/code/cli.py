from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset_loader import ECDatasetLoader


def run_phase1(args: argparse.Namespace):
    """Phase 1: Load dataset and export summary."""
    from .encoder import ECEncoder
    from .prototype_store import PrototypeStore
    from .export import export_metrics
    import json

    print(f"Loading dataset from: {args.artifact_dir}")
    loader = ECDatasetLoader(args.artifact_dir).load_all()

    output_path = Path(args.output_dir) / "dataset_summary.json"
    summary = loader.export_summary(output_path)

    encoder = ECEncoder()
    store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
    store.build_from_descriptions(encoder)

    proto_summary_path = Path(args.output_dir) / "prototype_summary.json"
    proto_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(proto_summary_path, "w", encoding="utf-8") as f:
        json.dump(store.summary(), f, indent=2)

    print(f"Phase 1 complete. Summary exported to {output_path}")
    print(f"Total rows: {summary['total_rows']} (Train: {summary['train_rows']}, Val: {summary['val_rows']}, Test: {summary['test_rows']})")
    print(f"Prototype summary: {store.summary()}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="EC-ProtoNet CLI")
    parser.add_argument("--artifact-dir", type=str, required=True, help="Path to dataset_builder output artifacts")
    parser.add_argument("--output-dir", type=str, default="protonet/output", help="Path to export results")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p1 = subparsers.add_parser("phase1", help="Run Phase 1: Loader and summary")
    p2 = subparsers.add_parser("ablation", help="Run EC-ProtoNet ablation study")
    p2.add_argument("--export-predictions", action="store_true", help="Export row-level prediction JSONL files")
    p3 = subparsers.add_parser("counterfactual", help="Run counterfactual consistency evaluation")
    
    args = parser.parse_args()
    
    if args.command == "phase1":
        sys.exit(run_phase1(args))
    elif args.command == "ablation":
        from .ablations import run_ablation_study
        export_preds = getattr(args, "export_predictions", False)
        run_ablation_study(args.artifact_dir, args.output_dir, export_preds=export_preds)
        sys.exit(0)
    elif args.command == "counterfactual":
        from .dataset_loader import ECDatasetLoader
        from .evaluator import ECEvaluator
        from .encoder import ECEncoder
        from .prototype_store import PrototypeStore
        from .scorer import ECProtoNetScorer
        from .config import ECConfig
        from .memory_support import MemorySupportModule
        
        loader = ECDatasetLoader(args.artifact_dir).load_all()
        pairs = loader.load_counterfactuals()
        
        encoder = ECEncoder()
        store = PrototypeStore().build_from_examples(loader.splits["train"], encoder)
        store.build_from_descriptions(encoder)
        
        memory_mod = MemorySupportModule.from_artifact(ECConfig(), Path(args.artifact_dir), encoder=encoder)
        store.build_from_memory(encoder, memory_mod.memory_summary)
        
        config = ECConfig(use_evidence=True, use_memory=True, use_novelty=True, use_contradiction=True, use_router=True)
        scorer = ECProtoNetScorer(encoder, store, config, memory_mod=memory_mod)
        
        evaluator = ECEvaluator()
        metrics = evaluator.evaluate_counterfactuals(pairs, scorer)
        
        output_path = Path(args.output_dir) / "counterfactual_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, indent=2)
            
        print(f"Counterfactual evaluation complete. Results: {metrics}")
        sys.exit(0)


if __name__ == "__main__":
    main()
