from __future__ import annotations

from .schema import ReviewExample, GoldInterpretation
from .config import ECConfig
import torch
import torch.nn.functional as F

class EvidenceSupportModule:
    def __init__(self, config: ECConfig, encoder=None):
        self.config = config
        self.encoder = encoder
        self.aspect_embs = {}
        self.query_cache = {}

    def get_support(self, query: ReviewExample, aspect: str) -> float:
        aspect_clean = aspect.lower().replace("_", " ")
        aspect_terms = aspect_clean.split()
        if not aspect_terms:
            return 0.0
            
        text_lower = query.text.lower()
        
        aspect_sim = 0.0
        if self.encoder is not None:
            if aspect_clean not in self.aspect_embs:
                with torch.no_grad():
                    self.aspect_embs[aspect_clean] = self.encoder.encode([aspect_clean])
            
            if text_lower not in self.query_cache:
                with torch.no_grad():
                    self.query_cache[text_lower] = self.encoder.encode([text_lower])
            
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
        
        scope_weight = 0.60
        if aspect_clean in text_lower:
            literal_match = 1.0
            scope_weight = 1.00
            
        trigger_similarity = aspect_sim
            
        evidence_support = (
            0.35 * aspect_sim +
            0.25 * trigger_similarity +
            0.20 * scope_weight +
            0.20 * literal_match
        )
        
        return min(1.0, max(0.0, evidence_support))
