from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def recreate(output_dir: Path) -> None:
    (output_dir / "episodic").mkdir(parents=True, exist_ok=True)
    (output_dir / "reviewlevel").mkdir(parents=True, exist_ok=True)


def _core_keep_set(input_dir: Path) -> set[Path]:
    return {
        input_dir / "episodic" / "train.jsonl",
        input_dir / "episodic" / "val.jsonl",
        input_dir / "episodic" / "test.jsonl",
        input_dir / "reviewlevel" / "train.jsonl",
        input_dir / "reviewlevel" / "val.jsonl",
        input_dir / "reviewlevel" / "test.jsonl",
    }


def _normalize_core_reviewlevel_names(input_dir: Path) -> None:
    reviewlevel_dir = input_dir / "reviewlevel"
    reviewlevel_dir.mkdir(parents=True, exist_ok=True)
    legacy_to_core = {
        "implicit_reviewlevel_train.jsonl": "train.jsonl",
        "implicit_reviewlevel_val.jsonl": "val.jsonl",
        "implicit_reviewlevel_test.jsonl": "test.jsonl",
    }
    for legacy_name, core_name in legacy_to_core.items():
        legacy = reviewlevel_dir / legacy_name
        core = reviewlevel_dir / core_name
        if legacy.exists() and not core.exists():
            legacy.rename(core)


def prune_input_jsonl(input_dir: Path) -> None:
    _normalize_core_reviewlevel_names(input_dir)
    keep = {p.resolve() for p in _core_keep_set(input_dir)}
    for path in input_dir.rglob("*.jsonl"):
        if path.resolve() in keep:
            continue
        path.unlink(missing_ok=True)
        print(f"Removed JSONL: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean ProtoBackend previous outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("ProtoBackend/outputs"))
    parser.add_argument("--input-dir", type=Path, default=Path("ProtoBackend/input"))
    parser.add_argument("--no-recreate", action="store_true", help="Do not recreate empty output folders.")
    parser.add_argument(
        "--no-prune-jsonl",
        action="store_true",
        help="Do not prune related input JSONL files.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        print(f"Removed: {args.output_dir}")
    else:
        print(f"Not found: {args.output_dir}")

    if not args.no_recreate:
        recreate(args.output_dir)
        print(f"Recreated clean output structure under: {args.output_dir}")

    if not args.no_prune_jsonl:
        prune_input_jsonl(args.input_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
