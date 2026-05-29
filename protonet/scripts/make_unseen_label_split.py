import argparse
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-labels", nargs="+", required=True)
    args = parser.parse_args()

    src = Path(args.artifact_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    
    holdouts = set(args.holdout_labels)
    removed_count = 0
    test_unseen_count = 0

    # 1. Process Train (Remove holdouts)
    with open(src / "train.jsonl", "r") as fin, open(dst / "train.jsonl", "w") as fout:
        for line in fin:
            row = json.loads(line)
            gold = row.get("gold_interpretations") or row.get("gold_aspects") or []
            
            row_labels = set()
            for g in gold:
                for key in ["aspect", "aspect_canonical", "aspect_raw"]:
                    if g.get(key):
                        row_labels.add(g.get(key))
            
            if any(label in holdouts for label in row_labels):
                removed_count += 1
                continue
            fout.write(line)

    # 2. Copy Val/Test (Keep all)
    shutil.copy(src / "val.jsonl", dst / "val.jsonl")
    shutil.copy(src / "test.jsonl", dst / "test.jsonl")
    
    # Check test for unseen rows
    with open(src / "test.jsonl", "r") as fin:
        for line in fin:
            row = json.loads(line)
            gold = row.get("gold_interpretations") or row.get("gold_aspects") or []
            
            row_labels = set()
            for g in gold:
                for key in ["aspect", "aspect_canonical", "aspect_raw"]:
                    if g.get(key):
                        row_labels.add(g.get(key))
            
            if any(label in holdouts for label in row_labels):
                test_unseen_count += 1
    # 3. Copy other metadata
    for f in ["manifest.json", "label_equivalence.json", "active_contract.json"]:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    manifest = {
        "holdout_labels": list(holdouts),
        "removed_from_train": removed_count,
        "test_unseen_rows": test_unseen_count,
        "status": "pass" if test_unseen_count > 0 else "fail_no_test_unseen"
    }
    
    with open(dst / "unseen_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Unseen split created at {dst}")
    print(f"Removed {removed_count} rows from train. Test contains {test_unseen_count} unseen rows.")

if __name__ == "__main__":
    main()
