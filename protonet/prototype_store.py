from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ECProtoNetV2Config
from .encoder import TextEncoder, cosine_matrix
from .schema import ReviewExample


@dataclass
class PrototypeStore:
    aspects: list[str]
    matrix: np.ndarray
    support_counts: dict[str, int]
    support_examples: dict[str, list[str]] = field(default_factory=dict)
    label_descriptions: dict[str, list[str]] = field(default_factory=dict)
    prototype_source: str = "train_evidence"


    def topk(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.matrix.size == 0 or query_vectors.size == 0:
            return np.zeros((query_vectors.shape[0], 0)), np.zeros((query_vectors.shape[0], 0), dtype=int)
        scores = cosine_matrix(query_vectors, self.matrix)
        k = min(k, scores.shape[1])
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(scores.shape[0])[:, None]
        partial = scores[row, idx]
        order = np.argsort(-partial, axis=1)
        idx_sorted = idx[row, order]
        score_sorted = scores[row, idx_sorted]
        return score_sorted.astype(np.float32), idx_sorted.astype(int)


def _support_text(example: ReviewExample, aspect: str, evidence_text: str, config: ECProtoNetV2Config) -> str:
    if config.include_aspect_name_in_support_text:
        return f"aspect: {aspect.replace('_', ' ')} evidence: {evidence_text or example.text}"
    return evidence_text or example.text


MAPPING_SCOPE_WEIGHTS = {
    "generic": 0.90,
    "domain_specific": 0.85,
    "generic+domain_specific": 1.00,
    "learned_store": 0.95,
    "open_world_candidate": 0.00,
    "provisional": 0.20,
    "unknown": 0.10,
    "": 0.10,
}


def _support_weight(example: ReviewExample, evidence_scope: str, source_type: str, mapping_scope: str, config: ECProtoNetV2Config) -> float:
    weight = 1.0
    if config.evidence_scope_weighting:
        weight *= float(config.evidence_scope_weights.get(evidence_scope or "unknown", 0.25))
    if config.source_type_weighting:
        weight *= float(config.source_type_weights.get(source_type or example.source_type or "unknown", 0.75))
    
    # Mapping scope weight
    weight *= MAPPING_SCOPE_WEIGHTS.get(mapping_scope or "", 0.10)
    
    return max(0.05, float(weight))


def build_prototype_store(
    examples: list[ReviewExample],
    encoder: TextEncoder,
    config: ECProtoNetV2Config,
    label_equivalence: dict[str, list[str]] | None = None,
) -> PrototypeStore:
    # support_by_aspect: list[tuple[text, weight, mapping_scope, label_type]]
    support_by_aspect: dict[str, list[tuple[str, float, str, str]]] = {}
    raw_support_examples: dict[str, list[str]] = {}

    for ex in examples:
        for g in ex.gold_aspects:
            if not g.aspect or g.aspect == "unknown":
                continue
            evidence_text = g.evidence_text if config.use_evidence_text_for_prototypes else ex.text
            text = _support_text(ex, g.aspect, evidence_text, config)
            
            mapping_scope = g.mapping_scope or ""
            label_type = g.label_type or ex.source_type or "unknown"
            
            weight = _support_weight(ex, g.evidence_scope, label_type, mapping_scope, config)
            
            support_by_aspect.setdefault(g.aspect, []).append((text, weight, mapping_scope, label_type))
            raw_support_examples.setdefault(g.aspect, []).append(evidence_text or ex.text)

    # Union of aspects from training data and equivalence map
    all_aspects = set(support_by_aspect.keys())
    if label_equivalence:
        all_aspects.update(label_equivalence.keys())
    
    aspects = sorted(
        a for a in all_aspects 
        if (
            len(support_by_aspect.get(a, [])) >= config.min_support_per_aspect 
            or (
                config.allow_singleton_equivalence_prototypes 
                and label_equivalence 
                and a in label_equivalence
            )
        )
    )

    # Label Filter (Phase 1 & 6)
    aspects = [a for a in aspects if _is_valid_known_prototype_label(a, support_by_aspect.get(a, []), config)]
    
    if not aspects:
        return PrototypeStore(aspects=[], matrix=np.zeros((0, 0), dtype=np.float32), support_counts={}, support_examples={})

    proto_vectors: list[np.ndarray] = []
    kept_aspects: list[str] = []
    support_counts: dict[str, int] = {}

    for aspect in aspects:
        # 1. Evidence centroid (if any)
        items = support_by_aspect.get(aspect, [])
        evidence_proto = None
        if items:
            texts = [t for t, _, _, _ in items]
            weights = np.array([w for _, w, _, _ in items], dtype=np.float32)
            vectors = encoder.encode(texts)

            # support-set denoising (Phase 6)
            if len(items) >= 4:
                centroid = np.average(vectors, axis=0, weights=weights)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                sims = np.matmul(vectors, centroid.astype(np.float32))
                # Remove bottom 20% by similarity
                cutoff = np.quantile(sims, 0.20)
                keep = sims >= cutoff
                if keep.any():
                    vectors = vectors[keep]
                    weights = weights[keep]

            evidence_proto = np.average(vectors, axis=0, weights=weights)
            norm = np.linalg.norm(evidence_proto)
            if norm > 0:
                evidence_proto = evidence_proto / norm

        # 2. Description centroid (if any)
        desc_proto = None
        if label_equivalence and aspect in label_equivalence:
            aliases = label_equivalence[aspect]
            desc_text = f"aspect: {aspect.replace('_', ' ')} aliases: {', '.join(aliases)}"
            desc_proto = encoder.encode([desc_text])[0]
            norm = np.linalg.norm(desc_proto)
            if norm > 0:
                desc_proto = desc_proto / norm

        # 3. Blend
        if evidence_proto is not None and desc_proto is not None:
            proto = (config.train_evidence_weight * evidence_proto + config.description_weight * desc_proto)
        elif evidence_proto is not None:
            proto = evidence_proto
        elif desc_proto is not None:
            proto = desc_proto
        else:
            continue

        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm
        
        proto_vectors.append(proto.astype(np.float32))
        kept_aspects.append(aspect)
        support_counts[aspect] = len(items)

    if not proto_vectors:
        return PrototypeStore(aspects=[], matrix=np.zeros((0, 0), dtype=np.float32), support_counts={}, support_examples={})

    matrix = np.vstack(proto_vectors).astype(np.float32)
    return PrototypeStore(
        aspects=kept_aspects,
        matrix=matrix,
        support_counts=support_counts,
        support_examples={a: raw_support_examples.get(a, []) for a in kept_aspects},
        label_descriptions=label_equivalence or {},
        prototype_source="hybrid_description_evidence",
    )



def _is_valid_known_prototype_label(
    aspect: str,
    items: list[tuple[str, float, str, str]],
    config: ECProtoNetV2Config,
) -> bool:
    if not aspect or aspect == "unknown":
        return False

    bad_exact = {
        "general",
        "thing",
        "item",
        "product",
        "something",
        "more_information",
        "purchased_this",
        "slowdown",
        "slow_down",
        "slow",
        "fast",
        "good",
        "bad",
        "nina", # Example from plan
    }

    if aspect.lower() in bad_exact:
        return False

    if any(ch.isdigit() for ch in aspect):
        return False

    # Punctuation artifacts check (walmart.com, etc)
    if any(ch in aspect for ch in ".:/?!@#$%^&*()"):
        return False

    months = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    }

    if aspect.lower() in months:
        return False

    if config.exclude_open_world_mapping_from_known_prototypes:
        if items and all(scope == "open_world_candidate" for _, _, scope, _ in items):
            return False

    return True

