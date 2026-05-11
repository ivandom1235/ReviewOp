from __future__ import annotations

from .schema import ReviewExample, GoldInterpretation
from .config import ECConfig
import torch
import torch.nn.functional as F

class EvidenceSupportModule:
    def __init__(self, config: ECConfig, encoder=None, store=None):
        self.config = config
        self.encoder = encoder
        self.store = store
        self.aspect_embs = {}
        self.query_cache = {}

    def get_support(self, query: ReviewExample, aspect: str) -> float:
        aspect_clean = aspect.lower().replace("_", " ")
        aspect_terms = aspect_clean.split()
        if not aspect_terms:
            return 0.0
            
        # Use evidence span if available (EC-P4)
        text_to_score = query.evidence_text or query.text
        text_lower = text_to_score.lower()
        scope = query.evidence_scope or "unknown"
        
        aspect_sim = 0.0
        if self.encoder is not None:
            if aspect_clean not in self.aspect_embs:
                with torch.no_grad():
                    self.aspect_embs[aspect_clean] = self.encoder.encode([aspect_clean])
            
            # Use separate cache key for text_lower to handle spans
            if text_lower not in self.query_cache:
                with torch.no_grad():
                    self.query_cache[text_lower] = self.encoder.encode([text_lower])[0]
            
            emb_query = self.query_cache[text_lower]
            emb_aspect = self.aspect_embs[aspect_clean]
            
            with torch.no_grad():
                sims = F.cosine_similarity(emb_query, emb_aspect)
                aspect_sim = max(0.0, sims.item())
                
        STOP_WORDS = {"of", "the", "a", "an", "is", "for", "with", "and", "in", "to"}
        meaningful_terms = [t for t in aspect_terms if t not in STOP_WORDS and len(t) > 2]
        if not meaningful_terms:
            meaningful_terms = aspect_terms
            
        match_count = sum(1 for t in meaningful_terms if t in text_lower)
        literal_match = match_count / len(meaningful_terms) if meaningful_terms else 0.0
        
        # Scope weighting (EC-P4)
        # Penalize full_review, reward exact/clause
        scope_weight = self.config.evidence_scope_weights.get(scope, 0.25)
        
        # Handle legacy or alternative naming
        if scope == "exact" and "exact_phrase" in self.config.evidence_scope_weights:
            scope_weight = self.config.evidence_scope_weights["exact_phrase"]
        
        if aspect_clean in text_lower:
            literal_match = 1.0
            # If literal match is in the text, we upgrade scope confidence
            scope_weight = max(scope_weight, 0.80)
            
        aspect_description_sim = aspect_sim
        trigger_similarity = aspect_sim
        
        if self.store and aspect in self.store.description_embeddings and self.encoder:
            emb_query = self.query_cache[text_lower]
            emb_desc = self.store.description_embeddings[aspect]
            with torch.no_grad():
                desc_sims = F.cosine_similarity(emb_query, emb_desc, dim=0)
                aspect_description_sim = max(0.0, desc_sims.item())
                
        if self.store and aspect in self.store.trigger_embeddings and self.encoder:
            emb_query = self.query_cache[text_lower]
            trigger_embs = self.store.trigger_embeddings[aspect]
            max_trig_sim = 0.0
            for t_emb in trigger_embs:
                with torch.no_grad():
                    tsim = F.cosine_similarity(emb_query, t_emb, dim=0).item()
                    if tsim > max_trig_sim:
                        max_trig_sim = tsim
            trigger_similarity = max_trig_sim
            
        evidence_support = (
            0.35 * aspect_description_sim +
            0.25 * trigger_similarity +
            0.20 * scope_weight +
            0.20 * literal_match
        )
        
        return min(1.0, max(0.0, evidence_support))
