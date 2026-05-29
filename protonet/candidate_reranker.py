from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .schema import CandidateScore, ReviewExample

GENERIC_LABELS = {"quality", "value", "performance", "usability"}

@dataclass
class CandidateReranker:
    model: Pipeline
    feature_names: list[str]

    def score(
        self, 
        candidate, 
        *, 
        source_type: str, 
        top1_aspect: str = "", 
        other_aspects: set[str] = None, 
        sibling_confusion_matrix: dict = None,
        other_candidates: list[Any] | None = None,
        label_descriptions: dict[str, list[str]] | None = None,
        review_text: str = ""
    ) -> float:
        x = np.array([
            candidate_features(
                candidate, 
                source_type, 
                self.feature_names, 
                top1_aspect=top1_aspect, 
                other_aspects=other_aspects, 
                sibling_confusion_matrix=sibling_confusion_matrix,
                other_candidates=other_candidates,
                label_descriptions=label_descriptions,
                review_text=review_text
            )
        ], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

def candidate_features(
    c, 
    source_type: str, 
    names: list[str], 
    top1_aspect: str = "", 
    other_aspects: set[str] = None, 
    sibling_confusion_matrix: dict = None,
    other_candidates: list[Any] | None = None,
    label_descriptions: dict[str, list[str]] | None = None,
    review_text: str = ""
) -> list[float]:
    from .import schema
    source = str(getattr(c, "candidate_source", "") or "")
    aspect = str(getattr(c, "aspect", "") or "")
    values = {
        "proto_score": float(getattr(c, "proto_score", 0.0) or 0.0),
        "final_score": float(getattr(c, "final_score", 0.0) or 0.0),
        "known_confidence": float(getattr(c, "known_confidence", 0.0) or 0.0),
        "lexical": float(getattr(c, "lexical_evidence_support", 0.0) or 0.0),
        "semantic": float(getattr(c, "semantic_evidence_support", 0.0) or 0.0),
        "description": float(getattr(c, "description_support", 0.0) or 0.0),
        "memory": float(getattr(c, "memory_support", 0.0) or 0.0),
        "margin": float(getattr(c, "margin_to_next", 0.0) or 0.0),
        "support_log": np.log1p(float(getattr(c, "support_count", 0) or 0)),
        "rank_inv": 1.0 / max(1.0, float(getattr(c, "rank", 1) or 1)),
        "unknown": float(getattr(c, "unknown_score", 0.0) or 0.0),
        "is_classifier": 1.0 if source == "classifier" else 0.0,
        "is_hybrid": 1.0 if source == "hybrid" else 0.0,
        "is_generic": 1.0 if aspect in GENERIC_LABELS else 0.0,
        "is_implicit": 1.0 if source_type in {"implicit", "synthetic", "counterfactual", "silver"} else 0.0,
    }

    # Hierarchical features (legacy fallback checks)
    PARENT_LABELS = {"quality", "service", "hardware", "experience", "system_experience", "value"}
    PARENT_LABELS_LIST = ["quality", "service", "hardware", "experience", "system_experience", "value"]
    
    values["is_parent_label"] = 1.0 if aspect in PARENT_LABELS else 0.0
    values["is_child_label"] = 1.0 if aspect in schema.ASPECT_PARENT else 0.0

    p_label = schema.parent_of(aspect)
    values["parent_label"] = float(PARENT_LABELS_LIST.index(p_label)) if p_label in PARENT_LABELS_LIST else -1.0

    # parent_child_conflict
    has_conflict = False
    if other_aspects:
        p_aspect = schema.parent_of(aspect)
        for oa in other_aspects:
            if oa != aspect:
                if schema.parent_of(oa) == aspect or oa == p_aspect:
                    has_conflict = True
                    break
    values["parent_child_conflict"] = 1.0 if has_conflict else 0.0

    # same_parent_as_top1
    if top1_aspect:
        values["same_parent_as_top1"] = 1.0 if schema.parent_of(aspect) == schema.parent_of(top1_aspect) else 0.0
    else:
        values["same_parent_as_top1"] = 0.0

    # sibling_confusion_score / risk
    sib_score = 0.0
    if sibling_confusion_matrix and other_aspects and aspect in sibling_confusion_matrix:
        matrix_for_aspect = sibling_confusion_matrix[aspect]
        vals = [matrix_for_aspect[oa] for oa in other_aspects if oa in matrix_for_aspect and oa != aspect]
        if vals:
            sib_score = max(vals)
    values["sibling_confusion_score"] = float(sib_score)
    values["sibling_confusion_risk"] = float(sib_score)

    # direct_evidence_for_child
    direct_child = 0.0
    if other_candidates and aspect in PARENT_LABELS:
        for oc in other_candidates:
            if oc.aspect != aspect and schema.parent_of(oc.aspect) == aspect:
                direct_child = max(direct_child, float(getattr(oc, "lexical_evidence_support", 0.0) or 0.0))
    values["direct_evidence_for_child"] = direct_child

    # direct_evidence_for_parent
    direct_parent = 0.0
    if other_candidates and aspect in schema.ASPECT_PARENT:
        p_aspect = schema.ASPECT_PARENT[aspect]
        for oc in other_candidates:
            if oc.aspect == p_aspect:
                direct_parent = max(direct_parent, float(getattr(oc, "lexical_evidence_support", 0.0) or 0.0))
    values["direct_evidence_for_parent"] = direct_parent

    # New graph-derived and profile features
    relation_val = 0.0
    if schema.ACTIVE_ASPECT_GRAPH is not None and top1_aspect:
        relation_val = schema.ACTIVE_ASPECT_GRAPH.get_relation_weight(aspect, top1_aspect)
    values["relation_to_top1"] = relation_val

    same_cluster = 0.0
    if schema.ACTIVE_ASPECT_GRAPH is not None and top1_aspect:
        hub_curr = schema.ACTIVE_ASPECT_GRAPH.cluster_of.get(aspect)
        hub_top = schema.ACTIVE_ASPECT_GRAPH.cluster_of.get(top1_aspect)
        if hub_curr and hub_top and hub_curr == hub_top:
            same_cluster = 1.0
    values["same_induced_cluster_as_top1"] = same_cluster

    cluster_conf = 0.0
    if schema.ACTIVE_ASPECT_GRAPH is not None and aspect in schema.ACTIVE_ASPECT_GRAPH.cluster_of:
        hub = schema.ACTIVE_ASPECT_GRAPH.cluster_of[aspect]
        cluster_conf = schema.ACTIVE_ASPECT_GRAPH.get_relation_weight(aspect, hub)
    values["cluster_confidence"] = cluster_conf

    overlap = 0.0
    if review_text and label_descriptions and aspect in label_descriptions:
        from .evidence_distilled_profiles import clean_and_tokenize
        tokens = set(clean_and_tokenize(review_text))
        profile_terms = set(label_descriptions[aspect])
        if profile_terms:
            overlap = len(tokens.intersection(profile_terms)) / len(profile_terms)
    values["evidence_profile_overlap"] = overlap

    values["distilled_profile_similarity"] = float(getattr(c, "description_support", 0.0) or 0.0)

    return [values[n] for n in names]

def train_candidate_reranker(rows, labels, sibling_confusion_matrix: dict = None, label_descriptions: dict[str, list[str]] | None = None) -> CandidateReranker:
    feature_names = [
        "proto_score", "final_score", "known_confidence", "lexical", "semantic",
        "description", "memory", "margin", "support_log", "rank_inv",
        "unknown", "is_classifier", "is_hybrid", "is_generic", "is_implicit",
        "parent_label", "is_parent_label", "is_child_label", "parent_child_conflict",
        "same_parent_as_top1", "sibling_confusion_score", "sibling_confusion_risk",
        "direct_evidence_for_child", "direct_evidence_for_parent",
        "relation_to_top1", "same_induced_cluster_as_top1", "cluster_confidence",
        "evidence_profile_overlap", "distilled_profile_similarity",
    ]
    X = []
    for row in rows:
        c = row[0]
        source_type = row[1]
        top1_aspect = row[2] if len(row) > 2 else ""
        other_aspects = row[3] if len(row) > 3 else None
        other_candidates = row[4] if len(row) > 4 else None
        review_text = row[5] if len(row) > 5 else ""
        feat = candidate_features(
            c, 
            source_type, 
            feature_names, 
            top1_aspect=top1_aspect, 
            other_aspects=other_aspects, 
            sibling_confusion_matrix=sibling_confusion_matrix,
            other_candidates=other_candidates,
            label_descriptions=label_descriptions,
            review_text=review_text
        )
        X.append(feat)

    X = np.array(X, dtype=float)
    y = np.array(labels, dtype=int)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear", random_state=42)),
    ])
    model.fit(X, y)
    return CandidateReranker(model=model, feature_names=feature_names)

def build_sibling_confusion_matrix(val_scores: dict[str, list[CandidateScore]], val_examples: list[ReviewExample]) -> dict[str, dict[str, float]]:
    from collections import defaultdict, Counter
    from .schema import parent_of
    
    # Map row_id to gold labels
    gold_by_row = {ex.row_id: set(ex.gold_labels) for ex in val_examples}
    
    fp_counts = Counter()
    confusion = defaultdict(Counter)
    
    for row_id, cands in val_scores.items():
        gold = gold_by_row.get(row_id, set())
        # We consider top-3 candidates as predicted to measure confusion
        pred = {c.aspect for c in cands[:3]}
        for fp in (pred - gold):
            fp_counts[fp] += 1
            for g in gold:
                if parent_of(fp) == parent_of(g) and fp != g:
                    confusion[fp][g] += 1
                    
    rates = defaultdict(dict)
    for fp, g_counts in confusion.items():
        total = fp_counts[fp]
        if total > 0:
            for g, count in g_counts.items():
                rates[fp][g] = count / total
    return rates
