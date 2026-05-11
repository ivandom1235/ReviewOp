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
        min_consistency: float = 0.75,
        min_evidence_quality: float = 0.75,
        max_contradiction: float = 0.20
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
        "soggy", "undercooked", "responsive", "frayed", "opened", "fray", "frays", "cut", "cutting", "died",
    }
    SENTIMENT_CUES = {"good", "bad", "great", "poor", "nice", "awful", "excellent", "terrible", "best", "worst", "love", "hate", "amazing", "horrible", "friendly"}
    BROAD_NOUNS = {"place", "thing", "service", "experience", "quality", "job", "work", "program", "feature", "item", "product", "stuff", "everything", "something", "area", "part"}

    def validate_for_review_queue(self, cluster_metrics: Dict[str, Any]) -> bool:
        """
        Promote clusters that represent repeated behavior patterns.
        """
        support = cluster_metrics.get("support_count", 0)
        reviews = cluster_metrics.get("unique_review_count", 0)
        consistency = cluster_metrics.get("cluster_consistency", 0.0)
        ev_quality = cluster_metrics.get("evidence_quality_mean", 0.0)
        contradiction = cluster_metrics.get("contradiction_score", 0.0)
        trigger_patterns = tuple(cluster_metrics.get("trigger_patterns", ()) or ())
        unique_surface_form_count = int(cluster_metrics.get("unique_surface_form_count", 0) or 0)
        source_types = set(cluster_metrics.get("source_types", ()) or ())
        
        # Phase 8: Adjust thresholds based on provenance
        # Explicit or previously learned patterns are more trusted
        is_trusted = bool(source_types & {"explicit", "implicit_learned", "implicit_json"})
        adj_min_consistency = self.min_consistency if not is_trusted else self.min_consistency - 0.05
        adj_min_ev_quality = self.min_evidence_quality if not is_trusted else self.min_evidence_quality - 0.10
        
        aspect_raw = str(cluster_metrics.get("aspect_raw", "")).lower().strip()
        if aspect_raw in {"unknown", "none", "null", "general", "misc"} and not trigger_patterns:
            return False
            
        if not trigger_patterns:
            return False

        trigger_tokens_list = [set(re.findall(r"\b\w+\b", str(p).lower())) for p in trigger_patterns if str(p).strip()]
        if not trigger_tokens_list:
            return False

        avg_trigger_len = sum(len(tokens) for tokens in trigger_tokens_list) / len(trigger_tokens_list)
        if avg_trigger_len < 2.5:
            return False

        behavior_hits = sum(1 for tokens in trigger_tokens_list if tokens & self.BEHAVIOR_CUES)
        sentiment_hits = sum(1 for tokens in trigger_tokens_list if tokens & self.SENTIMENT_CUES)
        entity_only_hits = sum(1 for tokens in trigger_tokens_list if len(tokens) <= 1 or (len(tokens) == 2 and all(len(token) <= 4 for token in tokens)))
        
        behavior_cue_rate = behavior_hits / len(trigger_tokens_list)
        sentiment_only_rate = sum(1 for tokens in trigger_tokens_list if tokens <= self.SENTIMENT_CUES) / len(trigger_tokens_list)
        entity_only_rate = entity_only_hits / len(trigger_tokens_list)
        
        # Reject if mostly noise
        if entity_only_rate > 0.20 and behavior_cue_rate < 0.5:
            return False
        if sentiment_only_rate > 0.30 and behavior_cue_rate < 0.5:
            return False
            
        # Reject broad nouns without behavior
        if aspect_raw in self.BROAD_NOUNS and behavior_cue_rate < 0.6:
            return False

        return (
            support >= self.min_support
            and reviews >= self.min_reviews
            and unique_surface_form_count >= self.min_surface_forms
            and consistency >= adj_min_consistency
            and ev_quality >= adj_min_ev_quality
            and contradiction <= self.max_contradiction
        )

