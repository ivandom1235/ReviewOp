from __future__ import annotations

from dataclasses import dataclass

from .config import ECProtoNetV2Config
from .schema import CandidateScore, RuntimeExample
from .text_features import looks_vague


@dataclass(frozen=True)
class RouterDecision:
    decision: str
    reason: str


class SelectiveRouterV2:
    def __init__(self, config: ECProtoNetV2Config):
        self.config = config

    def route_one(self, ex: RuntimeExample, cand: CandidateScore, is_top_candidate: bool = False) -> CandidateScore:
        cfg = self.config

        if not cand.aspect:
            return cand.with_decision("abstain", "empty_candidate")

        if is_top_candidate and looks_vague(ex.text) and cand.evidence_support < max(0.35, cfg.evidence_abstain_threshold):
            return cand.with_decision("abstain", "vague_text_low_evidence")

        if is_top_candidate and cand.final_score < cfg.abstain_threshold:
            return cand.with_decision("abstain", "below_abstain_threshold")

        if is_top_candidate and cand.evidence_support < cfg.evidence_abstain_threshold and cand.final_score < cfg.accept_threshold:
            return cand.with_decision("abstain", "low_evidence_and_low_score")

        if is_top_candidate and cand.proto_score < cfg.known_label_proto_floor and cand.evidence_support >= cfg.known_label_evidence_floor:
            return cand.with_decision("needs_review", "known_label_low_proto_review")

        if cand.margin_to_next < cfg.boundary_margin_threshold:
            return cand.with_decision("needs_review", "low_margin_boundary")

        if cand.final_score >= cfg.accept_threshold:
            return cand.with_decision("accept_known", "above_accept_threshold")

        return cand.with_decision("needs_review", "below_accept_threshold")

    def _make_open_world_candidate(
        self,
        ex: RuntimeExample,
        source: CandidateScore,
        reason: str,
    ) -> CandidateScore:
        from .text_features import open_world_evidence_quality
        quality = open_world_evidence_quality(ex.text)
        return CandidateScore(
            aspect="__open_world__",
            proto_score=0.0,
            final_score=quality,
            rank=1,
            evidence_support=quality,
            lexical_evidence_support=quality,
            semantic_evidence_support=0.0,
            memory_support=0.0,
            margin_to_next=source.margin_to_next,
            novelty_risk=source.novelty_risk,
            ambiguity_risk=source.ambiguity_risk,
            evidence_scope_score=source.evidence_scope_score,
            support_count=0,
            prototype_source="open_world_detector",
            candidate_type="open_world",
            source_aspect=source.aspect,
            known_confidence=source.known_confidence,
            open_world_evidence_quality=quality,
            residual_score=source.residual_score,
            energy_score=source.energy_score,
            unknown_score=source.unknown_score,
            decision="open_world_candidate",
            decision_reason=reason,
        )

    def _should_emit_open_world(
        self,
        ex: RuntimeExample,
        top: CandidateScore,
    ) -> bool:
        from .text_features import open_world_evidence_quality
        cfg = self.config
        if not cfg.emit_open_world_candidate:
            return False
        quality = open_world_evidence_quality(ex.text)
        if quality < cfg.open_world_evidence_quality_floor:
            return False
        if top.unknown_score < cfg.open_world_unknown_threshold:
            return False
        if top.known_confidence > cfg.open_world_known_confidence_ceiling:
            return False
        if top.proto_score > cfg.open_world_top1_proto_ceiling:
            return False
        if top.margin_to_next > cfg.open_world_margin_ceiling:
            return False
        if top.final_score >= cfg.accept_threshold:
            return False
        return True

    def route_candidates(self, ex: RuntimeExample, candidates: list[CandidateScore]) -> list[CandidateScore]:
        if not candidates:
            return [
                CandidateScore(
                    aspect="unknown",
                    proto_score=0.0,
                    final_score=0.0,
                    rank=1,
                    decision="abstain",
                    decision_reason="no_candidates",
                )
            ]

        routed: list[CandidateScore] = []
        for i, cand in enumerate(candidates):
            if i == 0:
                routed.append(self.route_one(ex, cand, is_top_candidate=True))
                continue
            
            threshold = self.config.class_thresholds.get(cand.aspect, self.config.accept_threshold)
            if (
                cand.final_score >= threshold
                and cand.evidence_support >= max(0.35, self.config.known_label_evidence_floor)
                and cand.proto_score >= self.config.known_label_proto_floor
            ):
                routed.append(cand.with_decision("accept_known", "multi_label_validated_accept"))
            else:
                routed.append(cand.with_decision("needs_review", "non_rank1_candidate_needs_review"))

        top = candidates[0]
        if self._should_emit_open_world(ex, top):
            open_cand = self._make_open_world_candidate(
                ex,
                top,
                "open_world_signal_emitted_without_blocking_known_labels",
            )
            return [open_cand] + routed

        return routed

