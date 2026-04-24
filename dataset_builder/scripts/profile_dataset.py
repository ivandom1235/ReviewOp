from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_builder.ingest.loaders import load_csv_reviews, load_jsonl_reviews
from dataset_builder.profile.dataset_profiler import profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    path = Path(args.input)
    rows = load_jsonl_reviews(path) if path.suffix.lower() == ".jsonl" else load_csv_reviews(path)
    print(json.dumps(profile_dataset(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
