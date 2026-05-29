import argparse
import json
from pathlib import Path
from dataclasses import replace

from protonet.config import ECProtoNetV2Config
from protonet.pipeline import run_compare
from protonet.io_utils import write_json

def main():
    parser = argparse.ArgumentParser(description="Run EC-ProtoNet configuration matrix evaluation")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Matrix configurations (Phase 3 & 7)
    configs = [
        {"encoder": "hashing", "model_name": None, "use_memory": False},
        {"encoder": "hashing", "model_name": None, "use_memory": True},
        {"encoder": "sentence-transformers", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "use_memory": False},
        {"encoder": "sentence-transformers", "model_name": "sentence-transformers/all-MiniLM-L6-v2", "use_memory": True},
        {"encoder": "sentence-transformers", "model_name": "sentence-transformers/all-mpnet-base-v2", "use_memory": False},
        {"encoder": "sentence-transformers", "model_name": "BAAI/bge-base-en-v1.5", "use_memory": False},
    ]
    
    matrix_results = []

    for c in configs:
        encoder = c["encoder"]
        model_name = c["model_name"]
        use_memory = c["use_memory"]
        run_name = f"{encoder}_{model_name.split('/')[-1] if model_name else 'default'}_memory_{'on' if use_memory else 'off'}"
        print(f"\n>>> Running configuration: {run_name}")
        
        run_output = output_base / run_name
        config = ECProtoNetV2Config(
            encoder=encoder,
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            use_memory=use_memory,
            require_artifact_pass=False,
            require_active_contract=False
        )

            
        try:
            result = run_compare(artifact_dir, run_output, config, split="test")
            metrics = result["metrics"]
            
            matrix_results.append({
                "run_name": run_name,
                "encoder": encoder,
                "use_memory": use_memory,
                "known_class_relaxed_f1": metrics["known_class_relaxed"]["f1"],
                "unseen_detection_f1": metrics["unseen_detection"]["f1"],
                "coverage": metrics["coverage"],
                "accepted_accuracy": metrics["accepted_accuracy_relaxed"]
            })
        except Exception as e:
            print(f"Error running {run_name}: {e}")


    # Write summary matrix
    write_json(output_base / "matrix_summary.json", matrix_results)
    
    # Print summary table (simple)
    print("\n" + "="*80)
    print(f"{'Run Name':<30} | {'Known F1':<10} | {'Unseen F1':<10} | {'Coverage':<10}")
    print("-" * 80)
    for r in matrix_results:
        print(f"{r['run_name']:<30} | {r['known_class_relaxed_f1']:<10.4f} | {r['unseen_detection_f1']:<10.4f} | {r['coverage']:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
