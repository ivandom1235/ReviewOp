from __future__ import annotations

import torch
import re
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

        self.evidence_mod = EvidenceSupportModule(config, encoder=encoder, store=store)
        self.memory_mod = memory_mod if memory_mod is not None else MemorySupportModule(config)
        self.novelty_mod = NoveltyModule(config)
        self.contradiction_mod = ContradictionModule(config)
        self.router = SelectiveRouter(config)

    def score_row(self, query: ReviewExample) -> list[ECScore]:
        """Compute full EC scores for a query row."""
        # Query strategy: Evidence-aware sentence selection (EC-P4 / Section 6.2.6)
        query_text = query.text
        
        if query.evidence_text:
            query_text = query.evidence_text
        elif len(query_text.split()) > 20:
            # Evidence-aware sentence selection (EC-P4 / Section 6.2.6)
            # Find the sentence that most likely contains the aspect evidence
            sentences = [s.strip() for s in re.split(r'[.!?]', query_text) if len(s.strip()) > 5]
            if sentences:
                sentence_embs = self.encoder.encode(sentences)
                max_sims = []
                for s_emb in sentence_embs:
                    # Find max similarity of this sentence to ANY prototype
                    s_sims = self.store.get_similarity(s_emb)
                    max_sims.append(max(s_sims.values()) if s_sims else 0.0)
                
                best_idx = int(torch.tensor(max_sims).argmax().item())
                query_text = sentences[best_idx]
            
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
        
        # Apply selective routing (EC-P4 / Section 6.2.5 bug fix)
        if self.config.use_router:
            scores = self.router.route_candidates(query, scores)
        
        return scores

    def predict(self, query: ReviewExample, top_k: int = 1) -> list[ECScore]:
        """Predict top-k aspects for a query row."""
        # Use score_row which already handles routing correctly
        scores = self.score_row(query)
        if not scores:
            return []
            
        return scores[:top_k]
