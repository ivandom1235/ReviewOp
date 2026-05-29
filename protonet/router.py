from __future__ import annotations

from dataclasses import dataclass

from .config import ECProtoNetV2Config
from .schema import CandidateScore, RuntimeExample
from .text_features import looks_vague

def suppress_confused_siblings(
    candidates: list[CandidateScore],
    config: ECProtoNetV2Config,
    sibling_confusion: dict[str, dict[str, float]] | None = None
) -> list[CandidateScore]:
    if not getattr(config, "use_sibling_confusion_suppression", False):
        return candidates
    from .import schema
    accepted_aspects = {c.aspect for c in candidates if c.decision == "accept_known" and c.aspect}
    aspect_to_cand = {c.aspect: c for c in candidates if c.aspect}
    suppressed = []
    direct_floor = getattr(config, "sibling_direct_evidence_floor", 0.45)
    graph_thresh = getattr(config, "graph_sibling_suppression_threshold", 0.45)
    for cand in candidates:
        if cand.decision == "accept_known" and cand.aspect:
            should_suppress = False
            triggering_sibling = ""

            # 1. Check validation-estimated sibling confusion
            if sibling_confusion and cand.aspect in sibling_confusion:
                cand_conf = sibling_confusion[cand.aspect]
                for sibling in accepted_aspects:
                    if sibling != cand.aspect and sibling in cand_conf:
                        if cand_conf[sibling] >= 0.60:
                            should_suppress = True
                            triggering_sibling = sibling
                            break

            # 2. Graph-derived sibling suppression
            if not should_suppress and schema.ACTIVE_ASPECT_GRAPH is not None:
                for sibling in accepted_aspects:
                    if sibling != cand.aspect:
                        w_rel = schema.ACTIVE_ASPECT_GRAPH.get_relation_weight(cand.aspect, sibling)
                        if w_rel >= graph_thresh:
                            sib_cand = aspect_to_cand.get(sibling)
                            if sib_cand is not None:
                                cand_lex = getattr(cand, "lexical_evidence_support", 0.0) or 0.0
                                sib_lex = getattr(sib_cand, "lexical_evidence_support", 0.0) or 0.0
                                cand_rank = getattr(cand, "rank", 1) or 1
                                sib_rank = getattr(sib_cand, "rank", 1) or 1
                                if cand_lex < sib_lex and cand_rank >= sib_rank:
                                    should_suppress = True
                                    triggering_sibling = sibling
                                    break

            if should_suppress:
                lex = getattr(cand, "lexical_evidence_support", 0.0) or 0.0
                if lex < direct_floor:
                    suppressed.append(cand.with_decision("needs_review", f"sibling_confusion_suppressed_by_{triggering_sibling}"))
                    continue
        suppressed.append(cand)
    return suppressed

def apply_candidate_reranker_gating(
    candidates: list[CandidateScore],
    config: ECProtoNetV2Config
) -> list[CandidateScore]:
    if not getattr(config, "use_candidate_reranker", False):
        return candidates
    reranker_threshold = getattr(config, "reranker_accept_threshold", 0.50)
    rescue_threshold = getattr(config, "recall_rescue_reranker_floor", 0.55)
    result = []
    for cand in candidates:
        if cand.decision == "accept_known":
            is_rescue = cand.decision_reason and ("rescue" in cand.decision_reason or "multi_label" in cand.decision_reason)
            threshold = rescue_threshold if is_rescue else reranker_threshold
            if cand.reranker_score < threshold:
                result.append(cand.with_decision("needs_review", "reranker_rejected"))
                continue
        result.append(cand)
    return result


@dataclass(frozen=True)
class RouterDecision:
    decision: str
    reason: str


def get_accept_threshold(aspect: str, config: ECProtoNetV2Config, source_type: str = "implicit") -> float:
    if config.class_thresholds and aspect in config.class_thresholds:
        return float(config.class_thresholds[aspect])
    is_implicit = str(source_type or "").lower() in {"implicit", "counterfactual", "synthetic", "silver"}
    if is_implicit:
        v = getattr(config, "implicit_accept_threshold", None)
        return float(config.accept_threshold if v is None else v)
    else:
        v = getattr(config, "explicit_accept_threshold", None)
        return float(config.accept_threshold if v is None else v)


def _classifier_candidate_is_grounded(candidate: CandidateScore, config: ECProtoNetV2Config) -> bool:
    lexical = getattr(candidate, "lexical_evidence_support", 0.0) or 0.0
    desc = getattr(candidate, "description_support", 0.0) or 0.0
    proto = getattr(candidate, "proto_score", 0.0) or 0.0
    known_conf = getattr(candidate, "known_confidence", 0.0) or 0.0

    if lexical > 0.0 or desc > 0.0:
        return True
    if proto > 0.0:
        return True
    high_floor = getattr(config, "classifier_accept_high_conf_floor", 0.55)
    if known_conf >= high_floor:
        return True
    return False


def limit_recall_rescue_accepts(candidates: list[CandidateScore], config: ECProtoNetV2Config) -> list[CandidateScore]:
    cap = getattr(config, "max_recall_rescue_accepts_per_review", 2)
    rescue_indices = []
    for idx, cand in enumerate(candidates):
        if cand.decision == "accept_known" and cand.decision_reason and "rescue_accept" in cand.decision_reason:
            rescue_indices.append(idx)
    if len(rescue_indices) <= cap:
        return candidates
    sorted_rescue_indices = sorted(rescue_indices, key=lambda i: candidates[i].final_score, reverse=True)
    demoted_indices = set(sorted_rescue_indices[cap:])
    result = []
    for idx, cand in enumerate(candidates):
        if idx in demoted_indices:
            result.append(cand.with_decision("needs_review", "rescue_accept_cap_exceeded"))
        else:
            result.append(cand)
    return result


def route_candidate(candidate: CandidateScore, config: ECProtoNetV2Config) -> CandidateScore:
    router = SelectiveRouterV2(config)
    ex = RuntimeExample(
        row_id="dummy",
        review_id="dummy",
        text="dummy text",
        domain="dummy",
        split="test",
        source_type="implicit",
    )
    return router.route_one(ex, candidate, is_top_candidate=True)


class SelectiveRouterV2:
    def __init__(self, config: ECProtoNetV2Config, sibling_confusion: dict | None = None):
        self.config = config
        self.sibling_confusion = sibling_confusion

    def _is_implicit_source(self, ex: RuntimeExample) -> bool:
        s = str(getattr(ex, "source_type", "") or "").lower()
        return s in {"implicit", "counterfactual", "synthetic", "silver"}

    def _accept_threshold(self, ex: RuntimeExample, aspect: str | None = None) -> float:
        if aspect is not None:
            return get_accept_threshold(aspect, self.config, getattr(ex, "source_type", "implicit"))
        if self._is_implicit_source(ex):
            v = getattr(self.config, "implicit_accept_threshold", None)
            return float(self.config.accept_threshold if v is None else v)
        v = getattr(self.config, "explicit_accept_threshold", None)
        return float(self.config.accept_threshold if v is None else v)

    def _known_evidence_floor(self, ex: RuntimeExample) -> float:
        if self._is_implicit_source(ex):
            v = getattr(self.config, "implicit_known_label_evidence_floor", None)
            return float(self.config.known_label_evidence_floor if v is None else v)
        v = getattr(self.config, "explicit_known_label_evidence_floor", None)
        return float(self.config.known_label_evidence_floor if v is None else v)

    def _can_accept_classifier_candidate(self, ex: RuntimeExample, cand: CandidateScore) -> bool:
        if not _classifier_candidate_is_grounded(cand, self.config):
            return False
        threshold = get_accept_threshold(cand.aspect, self.config, getattr(ex, "source_type", "implicit"))
        return (
            cand.candidate_source == "classifier"
            and cand.final_score >= threshold
            and cand.known_confidence >= 0.18
        )

    def route_one(self, ex: RuntimeExample, cand: CandidateScore, is_top_candidate: bool = False) -> CandidateScore:
        cfg = self.config
        accept_threshold = get_accept_threshold(cand.aspect, cfg, getattr(ex, "source_type", "implicit"))
        evidence_floor = self._known_evidence_floor(ex)

        if not cand.aspect:
            return cand.with_decision("abstain", "empty_candidate")

        if cand.candidate_source == "classifier" and not _classifier_candidate_is_grounded(cand, cfg):
            return cand.with_decision("needs_review", "classifier_candidate_low_grounding")

        if is_top_candidate and looks_vague(ex.text) and cand.evidence_support < max(0.35, cfg.evidence_abstain_threshold):
            return cand.with_decision("abstain", "vague_text_low_evidence")

        if is_top_candidate and cand.final_score < cfg.abstain_threshold:
            return cand.with_decision("abstain", "below_abstain_threshold")

        if is_top_candidate and cand.evidence_support < cfg.evidence_abstain_threshold and cand.final_score < accept_threshold:
            return cand.with_decision("abstain", "low_evidence_and_low_score")

        if self._can_accept_classifier_candidate(ex, cand):
            res = cand.with_decision("accept_known", "classifier_candidate_accept")
            if getattr(cfg, "use_candidate_reranker", False) and res.reranker_score < getattr(cfg, "reranker_accept_threshold", 0.50):
                return res.with_decision("needs_review", "reranker_rejected")
            return res

        if is_top_candidate and cand.proto_score < cfg.known_label_proto_floor and cand.evidence_support >= evidence_floor:
            return cand.with_decision("needs_review", "known_label_low_proto_review")

        if cand.margin_to_next < cfg.boundary_margin_threshold:
            return cand.with_decision("needs_review", "low_margin_boundary")

        if cand.final_score >= accept_threshold:
            res = cand.with_decision("accept_known", "above_accept_threshold")
            if getattr(cfg, "use_candidate_reranker", False) and res.reranker_score < getattr(cfg, "reranker_accept_threshold", 0.50):
                return res.with_decision("needs_review", "reranker_rejected")
            return res

        return cand.with_decision("needs_review", "below_accept_threshold")

    def _make_open_world_candidate(
        self,
        ex: RuntimeExample,
        source: CandidateScore,
        reason: str,
    ) -> CandidateScore:
        from .open_world_labeler import propose_open_world_label
        from .open_world_namer import infer_named_open_world_aspect
        from .text_features import open_world_evidence_quality
        quality = open_world_evidence_quality(ex.text)
        named = infer_named_open_world_aspect(
            ex.text,
            alias_map=getattr(self.config, "open_world_alias_map", None) or None,
            min_confidence=float(getattr(self.config, "open_world_name_min_confidence", 0.20)),
        )
        proposal = propose_open_world_label(ex.text)
        if named is not None:
            aspect = named.aspect
            final_score = max(float(quality), float(named.confidence))
            candidate_source = "named_open_world"
            decision = "named_open_world_candidate"
            decision_reason = named.reason
        else:
            aspect = proposal.label
            final_score = quality * float(proposal.confidence)
            candidate_source = f"open_world:{proposal.source}"
            decision = "open_world_candidate"
            decision_reason = reason
        return CandidateScore(
            aspect=aspect,
            proto_score=0.0,
            final_score=final_score,
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
            candidate_source=candidate_source,
            source_aspect=source.aspect,
            known_confidence=source.known_confidence,
            open_world_evidence_quality=quality,
            residual_score=source.residual_score,
            energy_score=source.energy_score,
            unknown_score=source.unknown_score,
            decision=decision,
            decision_reason=decision_reason,
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

            if cand.candidate_source == "classifier":
                if not _classifier_candidate_is_grounded(cand, self.config):
                    routed.append(cand.with_decision("needs_review", "classifier_candidate_low_grounding"))
                elif self._can_accept_classifier_candidate(ex, cand):
                    routed.append(cand.with_decision("accept_known", "classifier_rank_rescue_accept"))
                else:
                    routed.append(cand.with_decision("needs_review", "classifier_rank_candidate_needs_review"))
                continue

            if self._multi_label_accept(ex, cand):
                routed.append(cand.with_decision("accept_known", "recall_rescue_accept"))
                continue

            threshold = get_accept_threshold(cand.aspect, self.config, getattr(ex, "source_type", "implicit"))
            semantic_accept = (
                cand.final_score >= threshold
                and cand.known_confidence >= 0.28
                and cand.evidence_support >= max(0.16, self._known_evidence_floor(ex) * 0.45)
            )
            strong_proto_accept = (
                cand.proto_score >= self.config.known_label_proto_floor
                and cand.final_score >= threshold
            )
            if semantic_accept or strong_proto_accept:
                routed.append(cand.with_decision("accept_known", "multi_label_rank_rescue_accept"))
            else:
                routed.append(cand.with_decision("needs_review", "non_rank1_candidate_needs_review"))

        top = candidates[0]
        if self._should_emit_open_world(ex, top):
            open_cand = self._make_open_world_candidate(
                ex,
                top,
                "open_world_signal_emitted_without_blocking_known_labels",
            )
            routed = [open_cand] + routed

        # Apply candidate reranker gating if enabled
        routed = apply_candidate_reranker_gating(routed, self.config)

        # Apply sibling suppression if enabled
        routed = suppress_confused_siblings(routed, self.config, self.sibling_confusion)

        # Limit recall-rescue accepts before the total accepts cap
        routed = limit_recall_rescue_accepts(routed, self.config)

        max_accepts = max(1, int(getattr(self.config, "max_accepts_per_review", 3)))
        accept_count = 0
        capped: list[CandidateScore] = []
        for cand in routed:
            if cand.decision == "accept_known":
                accept_count += 1
                if accept_count > max_accepts:
                    capped.append(cand.with_decision("needs_review", "accept_cap_exceeded"))
                    continue
            capped.append(cand)
        return capped

    def _multi_label_accept(self, ex: RuntimeExample, cand: CandidateScore) -> bool:
        threshold = get_accept_threshold(cand.aspect, self.config, getattr(ex, "source_type", "implicit"))
        alias_ok = cand.lexical_evidence_support >= 0.25
        semantic_ok = cand.proto_score >= max(0.18, self.config.known_label_proto_floor * 0.70)
        classifier_ok = cand.candidate_source in {"classifier", "hybrid"} and cand.known_confidence >= 0.20
        score_ok = cand.final_score >= max(0.22, threshold * 0.85)

        if getattr(self.config, "use_candidate_reranker", False):
            reranker_floor = getattr(self.config, "recall_rescue_reranker_floor", 0.55)
            if cand.reranker_score < reranker_floor:
                return False

        return bool(score_ok and (alias_ok or semantic_ok or classifier_ok))

