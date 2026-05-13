import argparse
import json
import shutil
from pathlib import Path
from dataclasses import replace

from protonet.config import ECProtoNetV2Config
from protonet.dataset import load_dataset_bundle
from protonet.pipeline import run_compare
from protonet.io_utils import write_json, write_jsonl

def main():
    parser = argparse.ArgumentParser(description="EC-ProtoNet Closed-Loop Learning Study (Pass A -> Promotion -> Pass B)")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-labels", nargs="+", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    holdouts = set(args.holdout_labels)
    
    # --- PHASE 1: PASS A (No memory of holdouts) ---
    print("\n>>> PASS A: Initial evaluation (Memory Disabled)")
    pass_a_dir = output_base / "pass_a"
    config_a = ECProtoNetV2Config(
        encoder="sentence-transformers",
        accept_threshold=0.30,
        abstain_threshold=0.05,
        novel_threshold=0.80,
        boundary_margin_threshold=0.01,
        evidence_abstain_threshold=0.25,
        use_memory=False,
        emit_open_world_candidate=True,
        require_artifact_pass=False,
        require_active_contract=False
    )
    
    # We use the full artifact but disable memory
    # Note: The artifact should have holdouts removed from train prototypes (handled by make_unseen_split previously)
    results_a = run_compare(artifact_dir, pass_a_dir, config_a, split="test")
    
    # --- PHASE 2: PROMOTION (Expert Review Simulation) ---
    print("\n>>> PROMOTION: Simulating Expert Review and Memory Promotion")
    # We look at open-world candidates from Pass A and "promote" the ones that match holdouts
    promoted_entries = []
    
    open_world_file = pass_a_dir / "open_world_candidates.jsonl"
    if open_world_file.exists():
        with open(open_world_file, "r") as f:
            for line in f:
                r = json.loads(line)
                gold = set(r.get("gold_unseen_labels") or r.get("gold_labels") or [])
                # If any gold label is in our holdout set, simulate promoting it
                for label in gold:
                    if label in holdouts:
                        # Create a memory entry
                        promoted_entries.append({
                            "suggested_aspect": label,
                            "status": "promoted",
                            "representative_trigger": r.get("text"),
                            "quality": 1.0,
                            "support_count": 1
                        })
    
    print(f"Promoted {len(promoted_entries)} entries to AspectMemory.")
    
    # Create a temporary artifact for Pass B with the new memory
    pass_b_artifact = output_base / "temp_artifact_pass_b"
    shutil.copytree(artifact_dir, pass_b_artifact, dirs_exist_ok=True)
    write_json(pass_b_artifact / "aspect_memory_promoted.json", promoted_entries)
    
    # --- PHASE 3: PASS B (Memory Enabled) ---
    print("\n>>> PASS B: Evaluation with Promoted Memory")
    pass_b_dir = output_base / "pass_b"
    config_b = ECProtoNetV2Config(
        encoder="sentence-transformers",
        accept_threshold=0.30,
        abstain_threshold=0.05,
        novel_threshold=0.80,
        boundary_margin_threshold=0.01,
        evidence_abstain_threshold=0.25,
        use_memory=True,
        memory_min_status="promoted",
        emit_open_world_candidate=True,
        require_artifact_pass=False,
        require_active_contract=False
    )
    
    results_b = run_compare(pass_b_artifact, pass_b_dir, config_b, split="test")
    
    # --- PHASE 4: COMPARISON ---
    print("\n>>> RESULTS COMPARISON")
    f1_a = results_a["metrics"]["known_class_relaxed"]["f1"]
    f1_b = results_b["metrics"]["known_class_relaxed"]["f1"]
    
    unseen_f1_a = results_a["metrics"]["unseen_detection"]["f1"]
    unseen_f1_b = results_b["metrics"]["unseen_detection"]["f1"]
    
    print(f"Pass A Known F1: {f1_a:.4f}")
    print(f"Pass B Known F1: {f1_b:.4f}")
    print(f"Pass A Unseen F1: {unseen_f1_a:.4f}")
    print(f"Pass B Unseen F1: {unseen_f1_b:.4f}")
    
    improvement = f1_b - f1_a
    print(f"Learning Delta (Known F1): {improvement:+.4f}")
    
    summary = {
        "pass_a": results_a["metrics"],
        "pass_b": results_b["metrics"],
        "learning_delta_known_f1": improvement,
        "promoted_count": len(promoted_entries)
    }
    write_json(output_base / "study_summary.json", summary)
    print(f"\nStudy complete. Summary saved to {output_base / 'study_summary.json'}")

if __name__ == "__main__":
    main()
