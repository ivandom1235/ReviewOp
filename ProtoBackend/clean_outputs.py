from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def recreate(output_dir: Path) -> None:
    (output_dir / "episodic").mkdir(parents=True, exist_ok=True)
    (output_dir / "reviewlevel").mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean ProtoBackend previous outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("ProtoBackend/outputs"))
    parser.add_argument("--no-recreate", action="store_true", help="Do not recreate empty output folders.")
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        print(f"Removed: {args.output_dir}")
    else:
        print(f"Not found: {args.output_dir}")

    if not args.no_recreate:
        recreate(args.output_dir)
        print(f"Recreated clean output structure under: {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
