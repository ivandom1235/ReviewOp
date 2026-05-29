import argparse
import json
from pathlib import Path

from protonet.dataset import load_dataset_bundle
from protonet.encoder import build_encoder
from protonet.prototype_store import build_prototype_store
from protonet.config import ECProtoNetV2Config
from protonet.io_utils import write_json

def main():
    parser = argparse.ArgumentParser(description="Check EC-ProtoNet prototype inventory quality")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--fail-on-suspicious", action="store_true")
    args = parser.parse_args()

    config = ECProtoNetV2Config(min_support_per_aspect=args.min_support)
    bundle = load_dataset_bundle(args.artifact_dir)
    
    # We need an encoder to build the store, use hashing as it's lightweight
    encoder = build_encoder("hashing", model_name="", normalize=True, hashing_dim=config.hashing_dim, batch_size=32)

    store = build_prototype_store(bundle.train, encoder, config, label_equivalence=bundle.label_equivalence)
    
    aspects = store.aspects
    suspicious = []
    digit_labels = []
    
    for a in aspects:
        if any(ch.isdigit() for ch in a):
            digit_labels.append(a)
        # Check for punctuation or other junk
        if any(ch in a for ch in ".:/?!@#$%^&*()"):
            suspicious.append(a)

    singleton_count = sum(1 for a, count in store.support_counts.items() if count == 1)
    singleton_ratio = singleton_count / len(aspects) if aspects else 0.0

    report = {
        "status": "fail" if (args.fail_on_suspicious and (digit_labels or suspicious)) else "pass",
        "prototype_count_after_filter": len(aspects),
        "digit_label_count": len(digit_labels),
        "digit_labels": digit_labels,
        "suspicious_labels": suspicious,
        "singleton_count": singleton_count,
        "singleton_ratio": singleton_ratio,
    }

    write_json(Path(args.output), report)
    print(json.dumps(report, indent=2))

    if report["status"] == "fail":
        print("\n>>> INVENTORY CHECK FAILED: Suspicious labels detected.")
        exit(1)
    else:
        print("\n>>> INVENTORY CHECK PASSED.")

if __name__ == "__main__":
    main()
