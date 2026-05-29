from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit(artifact_dir: str | Path, min_strict_novel_rate: float = 0.05, max_strict_novel_rate: float = 0.15, min_review_queue: int = 3) -> dict:
    artifact_dir = Path(artifact_dir)
    rows = []
    for split in ["train", "val", "test"]:
        rows.extend(read_jsonl(artifact_dir / f"{split}.jsonl"))
    total = len(rows)
    novelty_counts = Counter()
    novel_aspects = Counter()
    boundary_aspects = Counter()
    for row in rows:
        gold = row.get("gold_interpretations") or []
        row_novelty = row.get("novelty_status")
        if row_novelty:
            novelty_counts[row_novelty] += 1
        for g in gold:
            status = g.get("novelty_status", row_novelty or "known")
            aspect = g.get("aspect_canonical") or g.get("aspect") or g.get("aspect_raw") or "unknown"
            if status == "novel":
                novel_aspects[aspect] += 1
            elif status == "boundary":
                boundary_aspects[aspect] += 1

    strict_novel_rows = novelty_counts.get("novel", 0)
    strict_novel_rate = strict_novel_rows / total if total else 0.0

    memory_summary = read_json(artifact_dir / "aspect_memory_summary.json", {})
    review_queue_count = int(memory_summary.get("review_queue_count", 0) or 0)
    organic_review_queue_count = int(memory_summary.get("organic_review_queue_count", review_queue_count) or 0)

    failed = []
    if strict_novel_rate < min_strict_novel_rate or strict_novel_rate > max_strict_novel_rate:
        failed.append(f"strict_novel_rate={strict_novel_rate:.4f}, expected {min_strict_novel_rate:.2f}-{max_strict_novel_rate:.2f}")
    if review_queue_count < min_review_queue:
        failed.append(f"review_queue_count={review_queue_count}, expected >= {min_review_queue}")
    if organic_review_queue_count < min_review_queue:
        failed.append(f"organic_review_queue_count={organic_review_queue_count}, expected >= {min_review_queue}")

    suggestions = []
    if strict_novel_rate < min_strict_novel_rate:
        need = int((min_strict_novel_rate * max(total, 1)) - strict_novel_rows) + 1
        suggestions.append(f"Add at least {need} verified strict-novel rows before next 500-row run.")
    if review_queue_count < min_review_queue or organic_review_queue_count < min_review_queue:
        suggestions.append("Add repeated evidence variants for at least three novel clusters: call_reliability, streaming_quality, fabric_fraying.")

    return {
        "total_rows": total,
        "novelty_counts": dict(novelty_counts),
        "strict_novel_rate": strict_novel_rate,
        "top_novel_aspects": novel_aspects.most_common(20),
        "top_boundary_aspects": boundary_aspects.most_common(20),
        "review_queue_count": review_queue_count,
        "organic_review_queue_count": organic_review_queue_count,
        "failed_checks": failed,
        "suggestions": suggestions,
        "status": "pass" if not failed else "fail",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    print(json.dumps(audit(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
