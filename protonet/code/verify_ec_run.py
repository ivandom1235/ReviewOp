import json
from pathlib import Path
import sys

REQUIRED_METRICS = {
    "strict_f1",
    "relaxed_multi_gold_f1",
    "coverage",
    "coverage_on_valid_gold_rows",
    "accepted_accuracy",
    "abstention_f1",
    "novel_f1",
    "decision_distribution",
    "metrics_schema_version",
}


def verify_run(output_dir: str):
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Directory {output_dir} does not exist.")
        sys.exit(1)
        
    required_files = [
        "run_metadata.json",
        "prototype_summary.json",
        "comparison_table.csv"
    ]
    
    with open(output_path / "run_metadata.json") as f:
        meta = json.load(f)
        
    # Check for prediction files for each model
    models = meta.get("models", [])
    for model in models:
        required_files.append(f"predictions_{model}.jsonl")
        required_files.append(f"{model}_metrics.json")
    
    for f in required_files:
        if not (output_path / f).exists():
            print(f"Missing required file: {f}")
            sys.exit(1)
            
    # Version check
    if meta.get("metrics_schema_version") != "ec_eval_v4":
        print(f"metrics_schema_version is {meta.get('metrics_schema_version')}, expected ec_eval_v4")
        sys.exit(1)
        
    # Status checks
    if meta.get("artifact_status") != "pass":
        print(f"artifact_status is {meta.get('artifact_status')}, expected pass")
        sys.exit(1)

    if meta.get("ready_for_protonet") is not True:
        print(f"ready_for_protonet is {meta.get('ready_for_protonet')}, expected true")
        sys.exit(1)
        
    if meta.get("stale_output_guard") != "pass":
        print(f"stale_output_guard is {meta.get('stale_output_guard')}, expected pass")
        sys.exit(1)

    # Coverage check in each metrics file
    for model in models:
        with open(output_path / f"{model}_metrics.json") as f:
            metrics = json.load(f)
            missing = REQUIRED_METRICS - set(metrics.keys())
            if missing:
                print(f"{model}_metrics.json missing metrics: {sorted(missing)}")
                sys.exit(1)
            coverage = metrics.get("coverage", 0.0)
            if coverage > 1.0:
                print(f"CRITICAL: Coverage for {model} is {coverage} > 1.0")
                sys.exit(1)
            valid_coverage = metrics.get("coverage_on_valid_gold_rows", 0.0)
            if valid_coverage > 1.0:
                print(f"CRITICAL: Valid coverage for {model} is {valid_coverage} > 1.0")
                sys.exit(1)

    with open(output_path / "prototype_summary.json") as f:
        proto_summary = json.load(f)
        sources = proto_summary.get("prototype_sources", {})
        # Note: These thresholds might vary by dataset, keeping them but making them warnings if appropriate
        if sources.get("generic_description", 0) <= 0:
            print("No generic_description prototypes found")
            sys.exit(1)
            
    print("Verification passed.")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_ec_run.py <output_dir>")
        sys.exit(1)
    verify_run(sys.argv[1])
