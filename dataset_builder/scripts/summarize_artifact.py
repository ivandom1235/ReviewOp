from __future__ import annotations

import argparse
import json

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_builder.scripts.validate_artifact import validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    print(json.dumps({"export_counts": validate_artifact(args.artifact_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
