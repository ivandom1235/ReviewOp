from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def recreate_structure(output_dir: Path) -> None:
    (output_dir / "reviewlevel").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodic").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean dataset_builder output artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_builder/output"))
    parser.add_argument("--no-recreate", action="store_true", help="Do not recreate output subfolders after cleanup.")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        print(f"Removed: {args.output_dir}")
    else:
        print(f"Output folder not found, nothing to remove: {args.output_dir}")

    if not args.no_recreate:
        recreate_structure(args.output_dir)
        print(f"Recreated clean structure under: {args.output_dir}")


if __name__ == "__main__":
    main()
