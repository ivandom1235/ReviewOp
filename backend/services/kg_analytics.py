# proto/backend/services/kg_analytics.py
from __future__ import annotations

from typing import List, Optional
import networkx as nx
from sqlalchemy.orm import Session
from sqlalchemy import desc

from models.tables import AspectNode, AspectEdge


def centrality_leaderboard(db: Session, limit: int = 20, domain: Optional[str] = None) -> List[dict]:
    q = db.query(AspectNode)
    if domain:
        q = q.filter(AspectNode.domain == domain)

    # MySQL-safe ordering: non-null first, then desc
    q = q.order_by((AspectNode.centrality == None).asc(), desc(AspectNode.centrality))  # noqa: E711
    rows = q.limit(limit).all()

    out = []
    for r in rows:
        out.append(
            {
                "aspect": r.aspect_cluster,
                "centrality": float(r.centrality or 0.0),
                "df": int(r.df or 0),
                "idf": float(r.idf or 0.0),
            }
        )
    return out


def edges(db: Session, limit: int = 200, domain: Optional[str] = None, edge_type: Optional[str] = None) -> List[dict]:
    q = db.query(AspectEdge)
    if domain:
        q = q.filter(AspectEdge.domain == domain)
    if edge_type:
        q = q.filter(AspectEdge.edge_type == edge_type)

    q = q.order_by(AspectEdge.weight.desc())
    rows = q.limit(limit).all()

    return [
        {"src": e.src_aspect, "dst": e.dst_aspect, "edge_type": e.edge_type, "weight": float(e.weight)}
        for e in rows
    ]


def communities(db: Session, domain: Optional[str] = None, edge_type: str = "cooccurrence", min_weight: float = 2.0) -> List[dict]:
    q = db.query(AspectEdge)
    if domain:
        q = q.filter(AspectEdge.domain == domain)
    q = q.filter(AspectEdge.edge_type == edge_type)
    rows = q.all()

    g = nx.Graph()
    for e in rows:
        w = float(e.weight or 0.0)
        if w < min_weight:
            continue
        g.add_edge(e.src_aspect, e.dst_aspect, weight=w)

    comms = []
    cid = 0
    for comp in nx.connected_components(g):
        nodes = sorted(list(comp))
        if len(nodes) < 2:
            continue
        comms.append({"community_id": cid, "aspects": nodes})
        cid += 1

    comms.sort(key=lambda x: len(x["aspects"]), reverse=True)
    return comms