from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import ECProtoNetV2Config
from .encoder import TextEncoder
from .memory import MemorySupportIndex
from .prototype_store import PrototypeStore
from .schema import CandidateScore, ReviewExample
from .text_features import evidence_support


@dataclass
class ScoringContext:
    encoder: TextEncoder
    prototypes: PrototypeStore
    memory: MemorySupportIndex | None
    config: ECProtoNetV2Config


def _score_candidate(
    ex: ReviewExample,
    query_vec: np.ndarray,
    aspect: str,
    proto_score: float,
    rank: int,
    next_proto_score: float,
    context: ScoringContext,
) -> CandidateScore:
    cfg = context.config
    evidence_text = ex.query_text or ex.evidence_text or ex.text
    lexical = evidence_support(evidence_text, aspect.replace("_", " "), matched_terms=ex.matched_terms)
    semantic = float(proto_score)
    candidate_evidence = max(lexical, semantic * 0.75)
    
    mem = 0.0
    if cfg.use_memory and context.memory is not None:
        mem = context.memory.support_for(query_vec, aspect, cfg.memory_similarity_threshold)

    # Description support (lexical overlap with aspect name and aliases)

    desc_terms = [aspect.replace("_", " ")]
    if aspect in context.prototypes.label_descriptions:
        desc_terms.extend(context.prototypes.label_descriptions[aspect])
    description_support = max([evidence_support(evidence_text, t, matched_terms=ex.matched_terms) for t in desc_terms])

    margin = float(proto_score - next_proto_score)
    ambiguity_risk = max(0.0, cfg.boundary_margin_threshold - margin) / max(cfg.boundary_margin_threshold, 1e-6)
    
    known_confidence = (
        0.55 * float(proto_score)
        + 0.20 * float(candidate_evidence)
        + 0.15 * float(max(0.0, margin))
        + 0.10 * float(mem)
    )
    novelty_risk = max(0.0, 1.0 - known_confidence)
    
    scope_score = 0.0
    for g in ex.gold_aspects:
        if g.aspect == aspect:
            scope_score = max(scope_score, cfg.evidence_scope_weights.get(g.evidence_scope, 0.25))
    
    w_memory = cfg.w_memory if (cfg.use_memory and context.memory is not None and getattr(context.memory, "entries", None)) else 0.0
    prior = cfg.aspect_priors.get(aspect, 0.0)
    
    final = (
        cfg.w_proto * float(proto_score)
        + cfg.w_evidence * float(candidate_evidence)
        + cfg.w_description * float(description_support)
        + w_memory * float(mem)
        + cfg.w_margin * float(max(0.0, margin))
        + cfg.w_scope * float(scope_score)
        + cfg.w_prior * float(prior)
    ) / max(1e-6, cfg.w_proto + cfg.w_evidence + cfg.w_description + w_memory + cfg.w_margin + cfg.w_scope + cfg.w_prior)

    return CandidateScore(
        aspect=aspect,
        proto_score=float(proto_score),
        final_score=float(final),
        rank=rank,
        evidence_support=float(candidate_evidence),
        lexical_evidence_support=float(lexical),
        semantic_evidence_support=float(semantic),
        memory_support=float(mem),
        margin_to_next=float(margin),
        novelty_risk=float(novelty_risk),
        known_confidence=float(known_confidence),
        ambiguity_risk=float(ambiguity_risk),
        evidence_scope_score=float(scope_score),
        support_count=int(context.prototypes.support_counts.get(aspect, 0)),
        prototype_source=context.prototypes.prototype_source,
        candidate_type="known",
    )


def score_examples(examples: list[ReviewExample], context: ScoringContext, top_k: int | None = None) -> dict[str, list[CandidateScore]]:
    top_k = int(top_k or context.config.top_k)
    texts = [ex.query_text or ex.text for ex in examples]
    q = context.encoder.encode(texts)
    
    # 1. Base Prototype Scores
    score_mat, idx_mat = context.prototypes.topk(q, top_k + 1)
    
    # 2. Memory Aspects (New potential classes)
    memory_aspects = []
    if context.memory:
        memory_aspects = sorted(list(set(context.memory.trigger_aspects)))
    
    out: dict[str, list[CandidateScore]] = {}
    for row_i, ex in enumerate(examples):
        cands: list[CandidateScore] = []
        if idx_mat.shape[1] == 0 and not memory_aspects:
            out[ex.row_id] = cands
            continue
            
        seen_aspects = set()
        usable = min(top_k, idx_mat.shape[1])
        for j in range(usable):
            idx = int(idx_mat[row_i, j])
            aspect = context.prototypes.aspects[idx]
            proto = float(score_mat[row_i, j])
            next_proto = float(score_mat[row_i, j + 1]) if j + 1 < score_mat.shape[1] else 0.0
            cands.append(_score_candidate(ex, q[row_i], aspect, proto, j + 1, next_proto, context))
            seen_aspects.add(aspect)
            
        # 3. Add memory-only aspects as candidates
        for aspect in memory_aspects:
            if aspect in seen_aspects:
                continue
            # Note: For memory-only aspects, proto_score is 0 (not in training set)
            cands.append(_score_candidate(ex, q[row_i], aspect, 0.0, len(cands) + 1, 0.0, context))
            
        # Re-sort candidates by final_score and update rank
        cands.sort(key=lambda x: x.final_score, reverse=True)
        for i, c in enumerate(cands):
            cands[i] = replace_candidate_rank(c, i + 1)
            
        out[ex.row_id] = cands[:top_k]
    return out


def replace_candidate_rank(c: CandidateScore, rank: int) -> CandidateScore:
    # Helper since CandidateScore is frozen
    return replace(c, rank=rank)
