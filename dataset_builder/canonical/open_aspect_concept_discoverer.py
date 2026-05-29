from __future__ import annotations
from typing import Any, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OpenAspectConceptDiscoverer:
    """
    Orchestrates the discovery of open-world aspects from raw extraction results.
    It focuses on evidence patterns rather than raw nouns.
    """
    def __init__(self, memory: Any):
        self.memory = memory
        from .evidence_clusterer import EvidenceClusterer
        from .cluster_validator import ClusterValidator
        from .cluster_labeler import ClusterLabeler
        
        self.clusterer = EvidenceClusterer()
        self.validator = ClusterValidator()
        self.labeler = ClusterLabeler()

    def process_candidate(
        self, 
        aspect_raw: str, 
        review_id: str, 
        evidence_text: str, 
        domain: str,
        **kwargs
    ) -> str:
        """
        Processes a raw candidate and returns a cluster_id.
        """
        cluster_id = self.memory.add_evidence(
            aspect_raw=aspect_raw,
            review_id=review_id,
            evidence_text=evidence_text,
            domain=domain,
            sentiment=kwargs.get("sentiment", "unknown"),
            run_id=kwargs.get("run_id"),
        )
        return cluster_id

    def _extract_trigger_pattern(self, aspect_raw: str, evidence_text: str) -> str:
        """
        Extracts a behavioral trigger pattern from the evidence.
        Example: "The portions were very small" -> "portions very small"
        """
        text = " ".join(str(evidence_text or "").lower().split())
        if not text:
            return ""

        if aspect_raw and aspect_raw.lower().strip() in text:
            return text

        for cue in ("broke", "broken", "fraying", "loose", "dropped", "slow", "cold", "hot", "small", "late", "weak", "soggy"):
            if cue in text:
                return text

        return ""
