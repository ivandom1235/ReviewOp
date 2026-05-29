from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_memory_sidecars_are_count_consistent() -> None:
    root = Path("dataset_builder/output")
    summary_path = root / "aspect_memory_summary.json"
    promoted_path = root / "aspect_memory_promoted.json"
    queue_path = root / "aspect_memory_review_queue.json"
    metrics_path = root / "metrics_summary.json"
    if not (summary_path.exists() and promoted_path.exists() and queue_path.exists() and metrics_path.exists()):
        return

    summary = _read_json(summary_path)
    promoted = _read_json(promoted_path)
    queue = _read_json(queue_path)
    metrics = _read_json(metrics_path)

    promoted_items = promoted.get("items", []) if isinstance(promoted, dict) else []
    queue_items = queue.get("items", []) if isinstance(queue, dict) else []
    promoted_total = len(promoted_items) if isinstance(promoted_items, list) else 0
    queue_total = len(queue_items) if isinstance(queue_items, list) else 0
    summary_total = int(summary.get("promoted_entries_total", summary.get("promoted_count", 0)) or 0)
    summary_queue_total = int(summary.get("review_queue_count", 0) or 0)
    metrics_total = int((metrics.get("aspect_memory") or {}).get("promoted_entries_total", 0) or 0)
    metrics_queue_total = int((metrics.get("aspect_memory") or {}).get("review_queue_count", 0) or 0)

    assert summary_total == promoted_total
    assert metrics_total == summary_total
    assert summary_queue_total == queue_total
    assert metrics_queue_total == summary_queue_total


def test_memory_lifecycle_ids_are_disjoint() -> None:
    root = Path("dataset_builder/output")
    queue_path = root / "aspect_memory_review_queue.json"
    promoted_path = root / "aspect_memory_promoted.json"
    if not (queue_path.exists() and promoted_path.exists()):
        return

    queue = _read_json(queue_path)
    promoted = _read_json(promoted_path)
    queue_items = queue.get("items", []) if isinstance(queue, dict) else []
    promoted_items = promoted.get("items", []) if isinstance(promoted, dict) else []

    queue_ids = {
        str(item.get("cluster_id"))
        for item in queue_items
        if isinstance(item, dict) and item.get("cluster_id")
    }
    promoted_ids = {
        str(item.get("cluster_id"))
        for item in promoted_items
        if isinstance(item, dict) and item.get("cluster_id")
    }
    assert queue_ids.isdisjoint(promoted_ids)
