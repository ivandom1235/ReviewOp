from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("review_queue") or payload.get("top_clusters") or []
        return list(items or [])
    return []


def main() -> None:
    p = argparse.ArgumentParser(description="Promote review-queue memory entries for controlled experiments.")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir)
    queue_path = artifact_dir / "aspect_memory_review_queue.json"
    promoted_path = artifact_dir / "aspect_memory_promoted.json"

    queue_items = _load_items(queue_path)
    selected = queue_items[: max(0, int(args.limit))]
    promoted_items = []
    for item in selected:
        promoted_items.append(
            {
                **item,
                "status": "promoted",
            }
        )

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total": len(promoted_items),
        "items": promoted_items,
    }
    promoted_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"promoted_written": len(promoted_items), "path": str(promoted_path)}, indent=2))


if __name__ == "__main__":
    main()
