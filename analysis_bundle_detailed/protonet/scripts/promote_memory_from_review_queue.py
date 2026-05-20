from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("items") or [])
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--min-quality", type=float, default=0.70)
    args = p.parse_args()

    artifact = Path(args.artifact_dir)
    queue_path = artifact / "aspect_memory_review_queue.json"
    out_path = artifact / "aspect_memory_promoted.json"
    items = _load_items(queue_path)

    promoted = []
    for item in items:
        support = int(item.get("support_count", item.get("unique_review_count", 0)) or 0)
        quality = float(item.get("quality", item.get("consistency", 0.0)) or 0.0)
        if support >= args.min_support and quality >= args.min_quality:
            promoted.append(
                {
                    **item,
                    "status": "promoted",
                    "validation_status": "auto_validated",
                    "promotion_policy": "train_val_only_min_support_quality",
                }
            )

    out_path.write_text(json.dumps({"total": len(promoted), "items": promoted}, indent=2), encoding="utf-8")
    print(json.dumps({"promoted": len(promoted)}, indent=2))


if __name__ == "__main__":
    main()
