from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from protonet.dataset import load_dataset_bundle
from protonet.io_utils import write_json


def _label_counts(rows) -> Counter[str]:
    c: Counter[str] = Counter()
    for ex in rows:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                c[g.aspect] += 1
    return c


def _bucket(count: int) -> str:
    if count >= 20:
        return "head"
    if count >= 5:
        return "torso"
    return "tail"


def _split_report(rows) -> dict:
    c = _label_counts(rows)
    buckets = {"head": 0, "torso": 0, "tail": 0}
    noisy_singletons: list[str] = []
    for label, n in c.items():
        buckets[_bucket(n)] += 1
        toks = [t for t in label.split("_") if t]
        if n == 1 and (len(toks) >= 3 or any(t in {"great", "excellent", "amazing", "perfect"} for t in toks)):
            noisy_singletons.append(label)
    return {
        "rows": len(rows),
        "unique_labels": len(c),
        "singleton_labels": sum(1 for _, n in c.items() if n == 1),
        "head_torso_tail": buckets,
        "top20": c.most_common(20),
        "noisy_singleton_candidates": sorted(noisy_singletons)[:100],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate label-space quality report.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    bundle = load_dataset_bundle(Path(args.artifact_dir))
    report = {
        "grouped": {
            "train": _split_report(bundle.splits["train"]),
            "val": _split_report(bundle.splits["val"]),
            "test": _split_report(bundle.splits["test"]),
        },
        "domain_holdout": {
            "train": _split_report(bundle.domain_holdout["train"]),
            "val": _split_report(bundle.domain_holdout["val"]),
            "test": _split_report(bundle.domain_holdout["test"]),
        },
    }
    write_json(Path(args.output), report)
    print(report)


if __name__ == "__main__":
    main()
