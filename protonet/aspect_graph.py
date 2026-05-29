from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from .encoder import TextEncoder
from .schema import ReviewExample

@dataclass
class AspectRelation:
    source: str
    target: str
    similarity: float
    cooccurrence: float
    val_confusion: float
    weight: float

@dataclass
class InducedAspectGraph:
    cluster_of: dict[str, str] = field(default_factory=dict)
    relations: list[AspectRelation] = field(default_factory=list)
    version: str = "v1"
    built_from_split: str = "train"
    relation_matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    support_counts: dict[str, int] = field(default_factory=dict)

    def get_relation_weight(self, a: str, b: str, encoder: TextEncoder | None = None) -> float:
        if a == b:
            return 1.0
        if (a, b) in self.relation_matrix:
            return self.relation_matrix[(a, b)]
        if (b, a) in self.relation_matrix:
            return self.relation_matrix[(b, a)]
        # Fallback to pure embedding similarity if we have the encoder
        if encoder is not None:
            lbl_a = f"aspect: {a.replace('_', ' ')}"
            lbl_b = f"aspect: {b.replace('_', ' ')}"
            v_a = encoder.encode([lbl_a])[0]
            v_b = encoder.encode([lbl_b])[0]
            norm_a = np.linalg.norm(v_a)
            norm_b = np.linalg.norm(v_b)
            if norm_a > 0:
                v_a /= norm_a
            if norm_b > 0:
                v_b /= norm_b
            return float(np.dot(v_a, v_b))
        return 0.0

def build_aspect_graph(
    train_rows: list[ReviewExample],
    encoder: TextEncoder,
    config: Any,
    validation_confusion: dict[str, dict[str, float]] | None = None,
) -> InducedAspectGraph:
    # 1. Gather all aspects present in training rows
    all_aspects = set()
    for ex in train_rows:
        for g in ex.gold_aspects:
            if g.aspect and g.aspect != "unknown":
                all_aspects.add(g.aspect)
    all_aspects = sorted(list(all_aspects))

    if not all_aspects:
        return InducedAspectGraph()

    # 2. Get representation for each aspect
    # r_a = alpha * Embed(a) + beta * Centroid(E_a) + gamma * Centroid(C_a)
    alpha = getattr(config, "aspect_graph_alpha", 0.50)
    beta = getattr(config, "aspect_graph_beta", 0.50)
    gamma = getattr(config, "aspect_graph_gamma", 0.00)

    # Pre-calculate Embed(a) for all a
    aspect_embeds = {}
    for a in all_aspects:
        lbl_text = f"aspect: {a.replace('_', ' ')}"
        val = encoder.encode([lbl_text])[0]
        norm = np.linalg.norm(val)
        if norm > 0:
            val = val / norm
        aspect_embeds[a] = val

    # Gather E_a and C_a
    E_a = {a: [] for a in all_aspects}
    C_a = {a: [] for a in all_aspects}
    support_counts = {a: 0 for a in all_aspects}

    for ex in train_rows:
        row_aspects = {g.aspect for g in ex.gold_aspects if g.aspect and g.aspect != "unknown"}
        for a in row_aspects:
            support_counts[a] += 1
            ev = next((g.evidence_text for g in ex.gold_aspects if g.aspect == a), "")
            E_a[a].append(ev or ex.text)
            C_a[a].append(ex.text)

    # Encode centroids
    r = {}
    for a in all_aspects:
        v_lbl = aspect_embeds[a]
        
        # Evidence centroid
        if E_a[a] and beta > 0.0:
            e_vecs = encoder.encode(E_a[a])
            v_ev = np.mean(e_vecs, axis=0)
            norm = np.linalg.norm(v_ev)
            if norm > 0:
                v_ev /= norm
        else:
            v_ev = np.zeros_like(v_lbl)

        # Context centroid
        if C_a[a] and gamma > 0.0:
            c_vecs = encoder.encode(C_a[a])
            v_ctx = np.mean(c_vecs, axis=0)
            norm = np.linalg.norm(v_ctx)
            if norm > 0:
                v_ctx /= norm
        else:
            v_ctx = np.zeros_like(v_lbl)

        # Combine
        v_ra = alpha * v_lbl + beta * v_ev + gamma * v_ctx
        norm = np.linalg.norm(v_ra)
        if norm > 0:
            v_ra /= norm
        r[a] = v_ra

    # 3. Compute PMI
    # Co-occurrence in train reviews
    N = len(train_rows)
    cooc = {}
    counts = {a: 0 for a in all_aspects}
    for ex in train_rows:
        row_aspects = {g.aspect for g in ex.gold_aspects if g.aspect and g.aspect != "unknown"}
        for a in row_aspects:
            counts[a] += 1
            for b in row_aspects:
                if a != b:
                    pair = tuple(sorted((a, b)))
                    cooc[pair] = cooc.get(pair, 0) + 1

    pmi = {}
    non_zero_pmis = []
    for i, a in enumerate(all_aspects):
        for b in all_aspects[i+1:]:
            pair = (a, b)
            if pair in cooc:
                p_ab = cooc[pair] / N
                p_a = counts[a] / N
                p_b = counts[b] / N
                val = math.log(p_ab / (p_a * p_b))
                pmi[pair] = val
                non_zero_pmis.append(val)
            else:
                pmi[pair] = 0.0

    # Scale PMI to [0, 1] for co-occurring pairs
    scaled_pmi = {}
    if non_zero_pmis:
        pmi_min = min(non_zero_pmis)
        pmi_max = max(non_zero_pmis)
        pmi_range = pmi_max - pmi_min
        for pair, val in pmi.items():
            if val > 0.0:
                if pmi_range > 1e-9:
                    scaled_pmi[pair] = (val - pmi_min) / pmi_range
                else:
                    scaled_pmi[pair] = 1.0
            else:
                scaled_pmi[pair] = 0.0
    else:
        for pair in pmi:
            scaled_pmi[pair] = 0.0

    # 4. Compute relation weights and edges
    lambda1 = getattr(config, "aspect_graph_lambda1", 0.50)
    lambda2 = getattr(config, "aspect_graph_lambda2", 0.30)
    lambda3 = getattr(config, "aspect_graph_lambda3", 0.20)

    # Adjust weights if validation_confusion is not provided or empty
    if not validation_confusion:
        tot = lambda1 + lambda2
        if tot > 0:
            lambda1 = lambda1 / tot
            lambda2 = lambda2 / tot
            lambda3 = 0.0
        else:
            lambda1 = 0.60
            lambda2 = 0.40
            lambda3 = 0.0

    relations = []
    relation_matrix = {}
    for i, a in enumerate(all_aspects):
        for b in all_aspects[i+1:]:
            pair = (a, b)
            cos_sim = float(np.dot(r[a], r[b]))
            pmi_val = scaled_pmi.get(pair, 0.0)
            conf_val = 0.0
            if validation_confusion:
                conf_val = max(
                    validation_confusion.get(a, {}).get(b, 0.0),
                    validation_confusion.get(b, {}).get(a, 0.0)
                )
            
            weight = lambda1 * cos_sim + lambda2 * pmi_val + lambda3 * conf_val
            
            rel = AspectRelation(
                source=a,
                target=b,
                similarity=cos_sim,
                cooccurrence=pmi_val,
                val_confusion=conf_val,
                weight=weight
            )
            relations.append(rel)
            relation_matrix[(a, b)] = weight
            relation_matrix[(b, a)] = weight

    # 5. Cluster the graph using single-linkage clustering
    theta = getattr(config, "aspect_graph_theta", 0.45)
    
    adj = {a: [] for a in all_aspects}
    for rel in relations:
        if rel.weight >= theta:
            adj[rel.source].append(rel.target)
            adj[rel.target].append(rel.source)

    visited = set()
    clusters = []
    for node in all_aspects:
        if node not in visited:
            comp = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(comp)

    # For each cluster, the hub is the aspect with the highest support count in training data
    cluster_of = {}
    for comp in clusters:
        hub = max(comp, key=lambda aspect: support_counts.get(aspect, 0))
        for aspect in comp:
            cluster_of[aspect] = hub

    return InducedAspectGraph(
        cluster_of=cluster_of,
        relations=relations,
        version="v2",
        built_from_split="train",
        relation_matrix=relation_matrix,
        support_counts=support_counts
    )
