import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    errors = {
        "gold_in_topk_not_accepted": 0,
        "gold_not_in_topk": 0,
        "known_as_open_world_fp": 0,
        "unseen_not_open_world_fn": 0,
        "abstain_false_positive": 0,
        "boundary_overaccept": 0
    }
    
    total_errors = 0
    
    with open(args.predictions, "r") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            
            gold = set(r.get("gold_labels", []))
            pred = set(r.get("predicted_labels", []))
            candidates = r.get("candidates", [])
            has_open_world = r.get("has_open_world", False)
            unseen = set(r.get("gold_unseen_labels", []))
            
            if pred != gold:
                total_errors += 1
                
                # 1. Gold in top-k but not accepted
                cand_aspects = [c["aspect"] for c in candidates]
                for g in gold:
                    if g in cand_aspects and g not in pred:
                        errors["gold_in_topk_not_accepted"] += 1
                
                # 2. Gold not in top-k
                for g in gold:
                    if g not in cand_aspects:
                        errors["gold_not_in_topk"] += 1
                
                # 3. Known as open-world FP
                if has_open_world and not unseen:
                    errors["known_as_open_world_fp"] += 1
                    
                # 4. Unseen not open-world FN
                if unseen and not has_open_world:
                    errors["unseen_not_open_world_fn"] += 1
                
                # 5. Abstain FP
                if r.get("has_abstain") and not r.get("abstain_acceptable"):
                    errors["abstain_false_positive"] += 1

    results = {
        "total_errors": total_errors,
        "error_counts": errors
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Error analysis saved to {args.output}")

if __name__ == "__main__":
    main()
