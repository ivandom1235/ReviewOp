# proto/backend/services/kg_build.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np
import networkx as nx
from sqlalchemy.orm import Session

from core.config import settings
from sentence_transformers import SentenceTransformer

from models.tables import Prediction, Review, AspectNode, AspectEdge


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
logger = logging.getLogger(__name__)
KG_BLOCKLIST = {"general", "unknown", "thing", "something", "everything", "means", "time", "place", "item"}
KG_MIN_QUALITY = 0.75


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _tokset(s: str) -> set[str]:
    return set(_WORD_RE.findall((s or "").lower()))


def _evidence_score(aspect: str, snippet: str) -> float:
    """
    0..1 overlap between aspect tokens and snippet tokens.
    """
    a = _tokset(aspect)
    if not a:
        return 0.0
    t = _tokset(snippet)
    inter = len(a & t)
    return float(inter / max(1, len(a)))


def _softmax(xs: List[float], tau: float = 2.5) -> List[float]:
    """
    Temperature-controlled softmax.
    tau > 1 => sharper (less uniform).
    """
    if not xs:
        return []
    z = np.asarray(xs, dtype=np.float64) * tau
    z = z - np.max(z)
    expz = np.exp(z)
    s = expz.sum()
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return (expz / s).tolist()


def _sentiment_to_num(sent: str) -> int:
    s = (sent or "").lower().strip()
    if s == "positive":
        return 1
    if s == "negative":
        return -1
    return 0


def _normalize_aspect(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join(s.split())


def _prediction_allowed_for_kg(pred: Prediction) -> bool:
    aspect = _normalize_aspect(pred.aspect_cluster or pred.aspect_raw or "")
    if not aspect or aspect in KG_BLOCKLIST:
        return False
    # Reuse existing field as optional quality carrier in this phase.
    if pred.aspect_weight is not None and float(pred.aspect_weight) < KG_MIN_QUALITY:
        return False
    return True


def _aspect_allowed_for_kg(aspect: str) -> bool:
    normalized = _normalize_aspect(aspect)
    return bool(normalized) and normalized not in KG_BLOCKLIST


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

@dataclass
class KGConfig:
    similarity_threshold: float = 0.75
    max_aspects_for_similarity: int = 5000
    similarity_top_k: int = 25
    alpha_evidence: float = 0.55
    beta_centrality: float = 0.25
    gamma_idf: float = 0.20
    overall_pos_thresh: float = 0.15
    overall_neg_thresh: float = -0.15


@lru_cache(maxsize=2)
def _cached_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, local_files_only=True)


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

class KGBuilder:

    def __init__(self, model_name: str = settings.kg_embedding_model_name):
        try:
            self.embedder = _cached_embedder(model_name)
            self.model_name = model_name
        except Exception:
            logger.exception("Failed to load KG embedder for model %s", model_name)
            self.embedder = None
            self.model_name = model_name

    # ---------------------------------------------------------
    # Main rebuild pipeline
    # ---------------------------------------------------------

    def rebuild(
        self,
        db: Session,
        domain: Optional[str] = None,
        cfg: Optional[KGConfig] = None,
    ) -> dict:

        cfg = cfg or KGConfig()

        # 1️⃣ Load predictions
        q = db.query(Prediction, Review).join(Review, Review.id == Prediction.review_id)
        if domain:
            q = q.filter(Review.domain == domain)
        rows = q.all()

        if not rows:
            return {"ok": True, "domain": domain, "message": "no predictions found"}

        by_review: Dict[int, List[Prediction]] = {}
        raw_aspects: List[str] = []

        for pred, rev in rows:
            if not _prediction_allowed_for_kg(pred):
                continue
            by_review.setdefault(pred.review_id, []).append(pred)
            raw_aspects.append(_normalize_aspect(pred.aspect_raw))

        unique_aspects = sorted(set(raw_aspects))
        if len(unique_aspects) > cfg.max_aspects_for_similarity:
            unique_aspects = unique_aspects[: cfg.max_aspects_for_similarity]

        # 2️⃣ Clear existing KG
        if domain:
            db.query(AspectEdge).filter(AspectEdge.domain == domain).delete(synchronize_session=False)
            db.query(AspectNode).filter(AspectNode.domain == domain).delete(synchronize_session=False)
        else:
            db.query(AspectEdge).delete(synchronize_session=False)
            db.query(AspectNode).delete(synchronize_session=False)

        db.flush()

        # 3️⃣ Similarity Graph
        sim_graph = self._build_similarity_graph(unique_aspects, cfg.similarity_threshold, top_k=cfg.similarity_top_k)

        # 4️⃣ Co-occurrence Graph
        co_graph = self._build_cooccurrence_graph(by_review)

        # 5️⃣ Clustering
        cluster_map = self._cluster_connected_components(sim_graph)

        # 6️⃣ Persist edges
        edge_counts = self._persist_edges(db, sim_graph, co_graph, cluster_map, domain)

        # 7️⃣ Update clusters
        updated = self._update_prediction_clusters(db, domain, cluster_map)

        # 8️⃣ Compute node stats (PageRank)
        node_stats = self._compute_and_persist_nodes(db, domain)

        # 9️⃣ Compute weights + overall sentiment
        weight_stats = self._compute_weights_and_overall(db, domain, cfg)

        db.commit()

        return {
            "ok": True,
            "domain": domain,
            "unique_aspects_raw": len(set(raw_aspects)),
            "unique_aspects_used_for_similarity": len(unique_aspects),
            "clusters": len(set(cluster_map.values())),
            "edges": edge_counts,
            "predictions_updated": updated,
            "nodes": node_stats,
            "weights": weight_stats,
        }

    # ---------------------------------------------------------
    # Graph Builders
    # ---------------------------------------------------------

    def build_for_review(self, review: Review) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Builds a small graph representing the aspects in a single review.
        """
        nodes = []
        node_map = {}
        aspects = []

        # 1. Build nodes from predictions
        for pred in getattr(review, "predictions", []) or []:
            a_raw = pred.aspect_raw
            a_cluster = pred.aspect_cluster or a_raw
            
            # Take first evidence span
            evidence_text = "-"
            ev_start = 0
            ev_end = 0
            if pred.evidence_spans:
                ev_obj = pred.evidence_spans[0]
                evidence_text = ev_obj.snippet
                ev_start = ev_obj.start_char
                ev_end = ev_obj.end_char

            node_id = f"single-{a_cluster.lower()}"
            if node_id in node_map:
                continue

            node = {
                "id": node_id,
                "label": a_cluster,
                "sentiment": pred.sentiment,
                "confidence": float(pred.confidence or 0.0),
                "evidence": evidence_text,
                "evidence_start": ev_start,
                "evidence_end": ev_end,
                "origin": "explicit", # Default to explicit
            }
            nodes.append(node)
            node_map[node_id] = node
            aspects.append(a_cluster)

        # 2. Build similarities if embedder is available
        edges = []
        if aspects and self.embedder:
            emb = self.embedder.encode(aspects, normalize_embeddings=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype=np.float32)
            sim = emb @ emb.T

            n_asp = len(aspects)
            for i in range(n_asp):
                for j in range(i + 1, n_asp):
                    w = float(sim[i, j])
                    if w >= 0.7: # Slightly lower threshold for single review
                        edges.append({
                            "source": f"single-{aspects[i].lower()}",
                            "target": f"single-{aspects[j].lower()}",
                            "type": "semantic",
                            "weight": w
                        })

        # 3. All-to-all co-occurrence within the review (with lower weight)
        for i in range(len(aspects)):
            for j in range(i + 1, len(aspects)):
                edges.append({
                    "source": f"single-{aspects[i].lower()}",
                    "target": f"single-{aspects[j].lower()}",
                    "type": "cooccurrence",
                    "weight": 0.5
                })

        return nodes, edges

    def _build_similarity_graph(self, aspects: List[str], threshold: float, top_k: int = 25) -> nx.Graph:
        g = nx.Graph()
        for a in aspects:
            g.add_node(a)

        if not aspects or self.embedder is None:
            return g

        emb = self.embedder.encode(aspects, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        sim = emb @ emb.T

        n = len(aspects)
        top_k = max(1, min(int(top_k or 25), max(1, n - 1)))
        for i in range(n):
            candidate_indices = np.argpartition(sim[i], -min(top_k + 1, n))[-min(top_k + 1, n):]
            for j in candidate_indices:
                if j <= i:
                    continue
                w = float(sim[i, j])
                if w >= threshold:
                    g.add_edge(aspects[i], aspects[j], weight=w)

        return g

    def _build_cooccurrence_graph(self, by_review: Dict[int, List[Prediction]]) -> Dict[Tuple[str, str], float]:
        counts = {}
        for rid, preds in by_review.items():
            aspects = sorted(set(_normalize_aspect(p.aspect_raw) for p in preds))
            for i in range(len(aspects)):
                for j in range(i + 1, len(aspects)):
                    a, b = sorted([aspects[i], aspects[j]])
                    counts[(a, b)] = counts.get((a, b), 0.0) + 1.0
        return counts

    def _cluster_connected_components(self, sim_graph: nx.Graph) -> Dict[str, str]:
        cluster_map = {}
        for comp in nx.connected_components(sim_graph):
            nodes = sorted(list(comp))
            canonical = min(nodes, key=lambda x: (len(x), x))
            for n in nodes:
                cluster_map[n] = canonical
        return cluster_map

    # ---------------------------------------------------------
    # Persist
    # ---------------------------------------------------------

    def _persist_edges(self, db, sim_graph, co_graph, cluster_map, domain):
        sim_edges = 0
        co_edges = 0

        for u, v, data in sim_graph.edges(data=True):
            cu = cluster_map.get(u, u)
            cv = cluster_map.get(v, v)
            if cu == cv:
                continue
            src, dst = sorted([cu, cv])
            db.add(AspectEdge(src_aspect=src, dst_aspect=dst,
                              edge_type="similarity",
                              weight=float(data.get("weight", 0.0)),
                              domain=domain))
            sim_edges += 1

        for (a, b), w in co_graph.items():
            ca = cluster_map.get(a, a)
            cb = cluster_map.get(b, b)
            if ca == cb:
                continue
            src, dst = sorted([ca, cb])
            db.add(AspectEdge(src_aspect=src, dst_aspect=dst,
                              edge_type="cooccurrence",
                              weight=float(w),
                              domain=domain))
            co_edges += 1

        db.flush()
        return {"similarity": sim_edges, "cooccurrence": co_edges}

    def _update_prediction_clusters(self, db, domain, cluster_map):
        q = db.query(Prediction, Review).join(Review, Review.id == Prediction.review_id)
        if domain:
            q = q.filter(Review.domain == domain)

        updated = 0
        for pred, rev in q.all():
            raw = _normalize_aspect(pred.aspect_raw)
            canon = cluster_map.get(raw, raw)
            if pred.aspect_cluster != canon:
                pred.aspect_cluster = canon
                updated += 1

        db.flush()
        return updated

    # ---------------------------------------------------------
    # Node Stats (PageRank centrality)
    # ---------------------------------------------------------

    def _compute_and_persist_nodes(self, db, domain):

        q = db.query(Prediction, Review).join(Review, Review.id == Prediction.review_id)
        if domain:
            q = q.filter(Review.domain == domain)
        rows = q.all()

        rq = db.query(Review)
        if domain:
            rq = rq.filter(Review.domain == domain)
        N = rq.count()

        aspect_to_reviews = {}
        for pred, _ in rows:
            if not _prediction_allowed_for_kg(pred):
                continue
            a = _normalize_aspect(pred.aspect_cluster)
            if not _aspect_allowed_for_kg(a):
                continue
            aspect_to_reviews.setdefault(a, set()).add(pred.review_id)

        eq = db.query(AspectEdge)
        if domain:
            eq = eq.filter(AspectEdge.domain == domain)

        g = nx.Graph()
        for e in eq.all():
            if e.edge_type != "cooccurrence":
                continue
            if e.weight > 0:
                g.add_edge(e.src_aspect, e.dst_aspect, weight=e.weight)

        pr = nx.pagerank(g, weight="weight") if g.number_of_nodes() > 0 else {}

        created = 0
        for a, rset in aspect_to_reviews.items():
            df = len(rset)
            idf = math.log((N + 1) / (df + 1)) if N else 0.0
            centrality = float(pr.get(a, 0.0))

            db.add(AspectNode(
                aspect_cluster=a,
                domain=domain,
                df=df,
                idf=idf,
                centrality=centrality
            ))
            created += 1

        db.flush()

        return {
            "N_reviews": N,
            "nodes_created": created,
            "graph_nodes": g.number_of_nodes(),
            "graph_edges": g.number_of_edges(),
        }

    # ---------------------------------------------------------
    # Weighting + Overall Sentiment
    # ---------------------------------------------------------

    def _compute_weights_and_overall(self, db, domain, cfg):

        nq = db.query(AspectNode)
        if domain:
            nq = nq.filter(AspectNode.domain == domain)

        node_map = {n.aspect_cluster: n for n in nq.all()}

        q = db.query(Review)
        if domain:
            q = q.filter(Review.domain == domain)

        updated_preds = 0
        updated_reviews = 0

        for rev in q.all():
            preds = rev.predictions or []
            if not preds:
                continue

            raw_scores = []

            for p in preds:
                if not _prediction_allowed_for_kg(p):
                    continue
                a = _normalize_aspect(p.aspect_cluster)
                if not _aspect_allowed_for_kg(a):
                    continue
                n = node_map.get(a)

                idf = n.idf if n else 0.0
                pr = n.centrality if n else 0.0

                snippet = p.evidence_spans[0].snippet if p.evidence_spans else ""
                evidence_score = _evidence_score(a, snippet)

                s = (
                    cfg.alpha_evidence * evidence_score
                    + cfg.beta_centrality * pr
                    + cfg.gamma_idf * (idf / (idf + 5.0))
                )

                raw_scores.append(max(0.0, float(s)))

            weights = _softmax(raw_scores, tau=2.5)

            overall_score = 0.0
            overall_conf = 0.0

            filtered_preds = [p for p in preds if _prediction_allowed_for_kg(p) and _aspect_allowed_for_kg(p.aspect_cluster)]
            for p, w in zip(filtered_preds, weights):
                score = _sentiment_to_num(p.sentiment) * (p.confidence or 0.0)
                p.aspect_weight = float(w)
                p.aspect_score = float(score)

                overall_score += w * score
                overall_conf += w * (p.confidence or 0.0)

                updated_preds += 1

            rev.overall_score = float(overall_score)
            rev.overall_confidence = float(overall_conf)

            if overall_score > cfg.overall_pos_thresh:
                rev.overall_sentiment = "positive"
            elif overall_score < cfg.overall_neg_thresh:
                rev.overall_sentiment = "negative"
            else:
                rev.overall_sentiment = "neutral"

            updated_reviews += 1

        db.flush()

        return {
            "predictions_scored": updated_preds,
            "reviews_scored": updated_reviews
        }
