from __future__ import annotations

import torch
from .schema import ReviewExample, ECScore
from .prototype_store import PrototypeStore
from .encoder import ECEncoder
from .config import ECConfig
from .evidence_support import EvidenceSupportModule
from .memory_support import MemorySupportModule
from .novelty import NoveltyModule
from .contradiction import ContradictionModule
from .selective_router import SelectiveRouter


class ECProtoNetScorer:
    def __init__(
        self,
        encoder: ECEncoder,
        store: PrototypeStore,
        config: ECConfig,
        memory_mod: MemorySupportModule | None = None,
    ):
        self.encoder = encoder
        self.store = store
        self.config = config

        self.evidence_mod = EvidenceSupportModule(config)
        self.memory_mod = memory_mod if memory_mod is not None else MemorySupportModule(config)
        self.novelty_mod = NoveltyModule(config)
        self.contradiction_mod = ContradictionModule(config)
        self.router = SelectiveRouter(config)

    def score_row(self, query: ReviewExample) -> list[ECScore]:
        """Compute full EC scores for a query row."""
        # Query strategy: try to use first sentence if full text is too long
        query_text = query.text
        if len(query_text.split()) > 20:
            # Simple heuristic: first sentence usually contains the main aspect
            query_text = query_text.split(".")[0]
            
        query_embedding = self.encoder.encode([query_text])[0]
        similarities = self.store.get_similarity(query_embedding)
        
        sim_list = list(similarities.values())
        
        scores = []
        for aspect, proto_sim in similarities.items():
            # Module signals
            ev_support = self.evidence_mod.get_support(query, aspect) if self.config.use_evidence else 0.0
            mem_support = self.memory_mod.get_support(query, aspect) if self.config.use_memory else 0.0
            nov_risk = self.novelty_mod.compute_risk(sim_list, mem_support) if self.config.use_novelty else 0.0
            contra_score = self.contradiction_mod.compute_penalty(query, aspect, ev_support) if self.config.use_contradiction else 0.0
            
            # Weighted aggregation
            final_score = (
                self.config.w_proto * proto_sim +
                self.config.w_evidence * ev_support +
                self.config.w_memory * mem_support -
                self.config.w_novelty * nov_risk -
                self.config.w_contradiction * contra_score
            )
            
            scores.append(
                ECScore(
                    aspect=aspect,
                    proto_similarity=proto_sim,
                    evidence_support=ev_support,
                    memory_support=mem_support,
                    novelty_risk=nov_risk,
                    contradiction_score=contra_score,
                    final_score=final_score,
                    prototype_source=self.store.prototype_sources.get(aspect),
                )
            )
        
        # Sort by score descending
        scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply selective routing
        if self.config.use_router:
            decision = self.router.route(query, scores)
            for s in scores:
                s = ECScore(**{**s.__dict__, "decision": decision}) # Simplified update
        
        return scores

    def predict(self, query: ReviewExample, top_k: int = 1) -> list[ECScore]:
        scores = self.score_row(query)
        if not scores:
            return []
            
        # For evaluation, we return the decision-aware scores
        results = []
        decision = self.router.route(query, scores) if self.config.use_router else "accept_known"
        
        for s in scores[:top_k]:
            results.append(
                ECScore(
                    aspect=s.aspect,
                    proto_similarity=s.proto_similarity,
                    evidence_support=s.evidence_support,
                    memory_support=s.memory_support,
                    novelty_risk=s.novelty_risk,
                    contradiction_score=s.contradiction_score,
                    final_score=s.final_score,
                    decision=decision
                )
            )
        return results
