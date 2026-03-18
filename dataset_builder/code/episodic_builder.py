"""Convert review-level rows to episodic rows."""
from __future__ import annotations

from typing import Dict, List


def build_episodic_rows(review_rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for row in review_rows:
        labels = row.get("labels", [])
        for idx, label in enumerate(labels, start=1):
            aspect = str(label.get("aspect", "")).strip()
            if not aspect:
                continue
            out.append(
                {
                    "example_id": f"{row['id']}_e{idx}",
                    "parent_review_id": row["id"],
                    "review_text": row.get("review_text", ""),
                    "evidence_sentence": label.get("evidence_sentence", ""),
                    "domain": row.get("domain", "generic"),
                    "aspect": aspect,
                    "implicit_aspect": aspect,
                    "sentiment": label.get("sentiment", "unknown"),
                    "label_type": label.get("type", "explicit"),
                    "split": row.get("split", "train"),
                    "source": row.get("source", "unknown"),
                }
            )
    return out
