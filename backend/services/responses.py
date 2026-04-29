from __future__ import annotations
from typing import List, Dict, Any
from models.schemas import InferReviewOut

class ContractMapper:
    """
    Deep module responsible for mapping internal pipeline results
    to the V6 API response models.
    """
    def to_infer_review_out(
        self,
        review_obj: Any,
        final_predictions: List[Dict[str, Any]],
        implicit_predictions: List[Dict[str, Any]]
    ) -> InferReviewOut:
        from models.schemas import (
            EvidenceSpanOut, 
            PredictionOut, 
            SelectivePredictionOut, 
            AbstainedPredictionOut, 
            NovelCandidateOut
        )
        from services.aspect_quality import apply_domain_gate_to_implicit_predictions
        from services.review_pipeline import split_selective_states

        # 1. Map Final Predictions
        preds_out: List[PredictionOut] = []
        for pred in final_predictions:
            spans = [
                EvidenceSpanOut(
                    start_char=int(ev.get("start_char", 0)),
                    end_char=int(ev.get("end_char", 0)),
                    snippet=str(ev.get("snippet", "")),
                )
                for ev in pred.get("evidence_spans", []) or []
            ]
            source = pred.get("source") or "verified"
            preds_out.append(
                PredictionOut(
                    aspect_raw=pred["aspect_raw"],
                    aspect_cluster=pred.get("aspect_cluster") or pred["aspect_raw"],
                    sentiment=pred.get("sentiment") or "neutral",
                    confidence=float(pred.get("confidence", 0.0)),
                    evidence_spans=spans,
                    rationale=pred.get("rationale") or "",
                    source=source,
                    is_implicit=(source == "implicit"),
                    verification_status="kept" if source in {"explicit", "implicit", "verified"} else None,
                    decision=pred.get("decision"),
                    routing=pred.get("routing"),
                    ambiguity_score=pred.get("ambiguity_score"),
                    novelty_score=pred.get("novelty_score"),
                )
            )

        # 2. Map Implicit Selective States
        selective_states = split_selective_states(
            apply_domain_gate_to_implicit_predictions(implicit_predictions, review_obj.domain)
        )
        
        accepted_out: List[SelectivePredictionOut] = []
        for row in selective_states.get("accepted_predictions", []):
            aspect = str(row.get("aspect_cluster") or row.get("aspect_raw") or row.get("aspect") or "").strip()
            if not aspect:
                continue
            
            # Extract first evidence span if possible
            evidence, start, end = None, None, None
            spans = row.get("evidence_spans") or []
            if spans:
                first = spans[0] or {}
                evidence = str(first.get("snippet") or "")
                start = int(first.get("start_char")) if first.get("start_char") is not None else None
                end = int(first.get("end_char")) if first.get("end_char") is not None else None

            accepted_out.append(
                SelectivePredictionOut(
                    aspect=aspect,
                    sentiment=str(row.get("sentiment") or "neutral"),
                    confidence=float(row.get("confidence", 0.0)),
                    routing=str(row.get("routing") or "known"),
                    evidence=evidence,
                    evidence_start=start,
                    evidence_end=end,
                )
            )

        abstained_out = [
            AbstainedPredictionOut(
                reason=str(row.get("reason") or "low_selective_confidence"),
                confidence=float(row.get("confidence", 0.0)),
                ambiguity_score=float(row.get("ambiguity_score", 0.0)),
            )
            for row in selective_states.get("abstained_predictions", [])
        ]

        novel_out = [
            NovelCandidateOut(
                aspect=str(row.get("aspect") or row.get("aspect_raw") or ""),
                novelty_score=float(row.get("novelty_score", 0.0)),
                confidence=float(row.get("confidence", 0.0)) if row.get("confidence") is not None else None,
            )
            for row in selective_states.get("novel_candidates", [])
            if str(row.get("aspect") or row.get("aspect_raw") or "").strip()
        ]

        return InferReviewOut(
            review_id=review_obj.id,
            domain=review_obj.domain,
            product_id=review_obj.product_id,
            predictions=preds_out,
            overall_sentiment=review_obj.overall_sentiment,
            overall_score=review_obj.overall_score,
            overall_confidence=review_obj.overall_confidence,
            accepted_predictions=accepted_out,
            abstained_predictions=abstained_out,
            novel_candidates=novel_out,
        )
