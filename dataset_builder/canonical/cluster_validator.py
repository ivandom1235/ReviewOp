from __future__ import annotations
from typing import Any, Dict
import re

class ClusterValidator:
    """
    Implements the quality gate for promoting evidence clusters in AspectMemory.
    """
    def __init__(
        self,
        min_support: int = 3,
        min_reviews: int = 3,
        min_surface_forms: int = 2,
        min_consistency: float = 0.70,
        min_evidence_quality: float = 0.70,
        max_contradiction: float = 0.25
    ):
        self.min_support = min_support
        self.min_reviews = min_reviews
        self.min_surface_forms = min_surface_forms
        self.min_consistency = min_consistency
        self.min_evidence_quality = min_evidence_quality
        self.max_contradiction = max_contradiction

    BEHAVIOR_CUES = {
        "broke", "broken", "fraying", "loose", "dropped", "dropping", "crashed", "waited",
        "waiting", "slow", "fast", "cold", "hot", "tiny", "small", "expensive", "cheap", "stale",
        "late", "delayed", "disconnected", "logged", "logging", "logout", "noisy", "buffering", "weak",
        "soggy", "undercooked", "responsive", "friendly", "terrible", "great", "poor", "frayed",
        "opened", "fray", "frays", "cut", "cutting", "died",
    }

    def validate_for_review_queue(self, cluster_metrics: Dict[str, Any]) -> bool:
        """
        New rule: Memory validity depends on evidence-cluster quality, not generic_parent.
        """
        support = cluster_metrics.get("support_count", 0)
        reviews = cluster_metrics.get("unique_review_count", 0)
        consistency = cluster_metrics.get("cluster_consistency", 0.0)
        ev_quality = cluster_metrics.get("evidence_quality_mean", 0.0)
        contradiction = cluster_metrics.get("contradiction_score", 0.0)
        trigger_patterns = tuple(cluster_metrics.get("trigger_patterns", ()) or ())
        unique_surface_form_count = int(cluster_metrics.get("unique_surface_form_count", 0) or 0)
        
        # Hard Rejects (Phase 3)
        aspect_raw = str(cluster_metrics.get("aspect_raw", "")).lower().strip()
        if aspect_raw in {"unknown", "none", "null", "general", "misc"}:
            return False
        if not trigger_patterns:
            return False
        avg_trigger_len = sum(len(str(p).split()) for p in trigger_patterns if str(p).strip()) / max(1, len(trigger_patterns))
        if avg_trigger_len < 3:
            return False
        if unique_surface_form_count < self.min_surface_forms:
            return False

        trigger_tokens = [set(re.findall(r"\b\w+\b", str(p).lower())) for p in trigger_patterns if str(p).strip()]
        behavior_hits = sum(1 for tokens in trigger_tokens if tokens & self.BEHAVIOR_CUES)
        entity_only_hits = sum(1 for tokens in trigger_tokens if len(tokens) <= 1 or (len(tokens) == 2 and all(len(token) <= 4 for token in tokens)))
        behavior_cue_rate = behavior_hits / max(1, len(trigger_tokens))
        entity_only_rate = entity_only_hits / max(1, len(trigger_tokens))
        if entity_only_rate > 0.20:
            return False
        if behavior_cue_rate <= 0.0 and ev_quality < 0.80:
            return False
            
        # Broad noun check - should ideally be handled by trigger patterns
        if len(aspect_raw.split()) == 1 and reviews < 5:
            # Broad nouns need more support to prove they aren't noise
            if aspect_raw in {"thing", "item", "product", "stuff"}:
                return False

        return (
            support >= self.min_support
            and reviews >= self.min_reviews
            and consistency >= self.min_consistency
            and ev_quality >= self.min_evidence_quality
            and contradiction <= self.max_contradiction
        )
