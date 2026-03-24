from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(raw: Path) -> Path:
    if raw.is_absolute():
        return raw
    if len(raw.parts) >= 2 and raw.parts[0] == "dataset_builder":
        return PROJECT_ROOT / Path(*raw.parts[1:])
    return PROJECT_ROOT / raw


def recreate_structure(output_dir: Path) -> None:
    (output_dir / "reviewlevel").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodic").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean dataset_builder output artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--no-recreate", action="store_true", help="Do not recreate output subfolders after cleanup.")
    args = parser.parse_args()

    output_dir = _resolve_project_path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Removed: {output_dir}")
    else:
        print(f"Output folder not found, nothing to remove: {output_dir}")

    if not args.no_recreate:
        recreate_structure(output_dir)
        print(f"Recreated clean structure under: {output_dir}")


if __name__ == "__main__":
    main()
