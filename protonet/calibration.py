from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from .config import ECProtoNetV2Config
from .dataset import DatasetBundle
from .evaluator import evaluate_predictions, objective
from .pipeline import build_context
from .router import SelectiveRouterV2
from .schema import PredictionRecord, to_runtime_example
from .scorer import score_examples


def grid_search_router(bundle: DatasetBundle, base_config: ECProtoNetV2Config) -> dict[str, Any]:
    accept_grid = [0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46]
    abstain_grid = [0.06, 0.10, 0.14, 0.18]
    unknown_grid = [0.50, 0.60, 0.70, 0.80, 0.90]
    margin_grid = [0.015, 0.025, 0.035, 0.050]
    evidence_grid = [0.10, 0.18, 0.25, 0.35]
    open_world_ceil_grid = [0.25, 0.30, 0.35, 0.40]
    open_world_qual_grid = [0.35, 0.45, 0.55]

    best: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []

    base_context = build_context(bundle, base_config)
    raw_scores = score_examples([to_runtime_example(ex) for ex in bundle.val], base_context, top_k=base_config.top_k)

    for accept_t, abstain_t, unknown_t, margin_t, evidence_t, ow_ceil, ow_qual in product(
        accept_grid, abstain_grid, unknown_grid, margin_grid, evidence_grid, open_world_ceil_grid, open_world_qual_grid
    ):
        if abstain_t >= accept_t:
            continue
        cfg = replace(
            base_config,
            accept_threshold=accept_t,
            abstain_threshold=abstain_t,
            open_world_unknown_threshold=unknown_t,
            boundary_margin_threshold=margin_t,
            evidence_abstain_threshold=evidence_t,
            open_world_known_confidence_ceiling=ow_ceil,
            open_world_evidence_quality_floor=ow_qual,
            require_active_contract=False,
        )

        router = SelectiveRouterV2(cfg)
        records: list[PredictionRecord] = []
        for ex in bundle.val:
            routed = router.route_candidates(ex, raw_scores.get(ex.row_id, []))
            records.append(
                PredictionRecord(
                    row_id=ex.row_id,
                    review_id=ex.review_id,
                    split=ex.split,
                    domain=ex.domain,
                    text=ex.text,
                    gold_labels=sorted(ex.gold_labels),
                    gold_novel_labels=sorted(ex.gold_novel_labels),
                    gold_boundary_labels=sorted(ex.gold_boundary_labels),
                    gold_emerging_labels=list(ex.gold_emerging_labels),
                    gold_unseen_labels=ex.gold_unseen_labels,
                    abstain_acceptable=ex.abstain_acceptable,
                    candidates=routed,
                )
            )
        metrics = evaluate_predictions(records, bundle.normalizer, cfg)
        score = objective(metrics)
        row = {
            "objective": score,
            "config": {
                "accept_threshold": accept_t,
                "abstain_threshold": abstain_t,
                "open_world_unknown_threshold": unknown_t,
                "boundary_margin_threshold": margin_t,
                "evidence_abstain_threshold": evidence_t,
                "open_world_known_confidence_ceiling": ow_ceil,
                "open_world_evidence_quality_floor": ow_qual,
            },
            "metrics": metrics,
        }
        results.append(row)
        if best is None or score > best["objective"]:
            best = row

    results.sort(key=lambda x: x["objective"], reverse=True)
    return {"best": best, "top10": results[:10], "trials": len(results)}
