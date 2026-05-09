from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ECScore, ReviewExample


def export_predictions(
    predictions: list[list[ECScore]],
    examples: list[ReviewExample],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for ex, preds in zip(examples, predictions):
            gold_aspects = [g.aspect for g in ex.gold_interpretations]
            row = {
                "row_id": ex.row_id,
                "review_id": ex.review_id,
                "domain": ex.domain,
                "text": ex.text,
                "gold_aspects": gold_aspects,
                "predictions": [
                    {
                        "aspect": p.aspect,
                        "decision": p.decision,
                        "final_score": round(p.final_score, 6),
                        "proto_similarity": round(p.proto_similarity, 6),
                        "evidence_support": round(p.evidence_support, 6),
                        "memory_support": round(p.memory_support, 6),
                        "novelty_risk": round(p.novelty_risk, 6),
                        "contradiction_score": round(p.contradiction_score, 6),
                        "prototype_source": p.prototype_source,
                    }
                    for p in preds
                ],
            }
            f.write(json.dumps(row) + "\n")


def export_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
