from __future__ import annotations

from .schema import ReviewExample
from .config import ECConfig
import json
from pathlib import Path
import torch
import torch.nn.functional as F

class MemorySupportModule:
    def __init__(self, config: ECConfig, memory_summary: dict | None = None, encoder=None):
        self.config = config
        self.memory_summary = memory_summary or {}
        self.encoder = encoder
        self.query_cache = {}
        self.trigger_embs_cache = {}
        
        if self.encoder is not None:
            top_clusters = self.memory_summary.get("top_clusters", [])
            for i, entry in enumerate(top_clusters):
                trigger_patterns = entry.get("trigger_patterns", [])
                representative = entry.get("representative_trigger")
                if representative and representative not in trigger_patterns:
                    trigger_patterns.append(representative)
                
                if trigger_patterns:
                    with torch.no_grad():
                        self.trigger_embs_cache[i] = self.encoder.encode(trigger_patterns)

    @classmethod
    def from_artifact(cls, config: ECConfig, artifact_dir: Path, encoder=None) -> MemorySupportModule:
        path = artifact_dir / "aspect_memory_summary.json"
        summary = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                
        queue_path = artifact_dir / "aspect_memory_review_queue.json"
        if queue_path.exists():
            with open(queue_path, "r", encoding="utf-8") as f:
                queue_data = json.load(f)
                queue_items = queue_data.get("items", [])
                
                # Merge into summary.top_clusters if not already there
                if "top_clusters" not in summary:
                    summary["top_clusters"] = []
                
                existing_ids = {c.get("cluster_id") for c in summary["top_clusters"]}
                for item in queue_items:
                    if item.get("cluster_id") not in existing_ids:
                        summary["top_clusters"].append(item)
                        
        # Filter for promoted only to ensure honesty in "learned memory" claims (EC-P4 fix)
        if "top_clusters" in summary:
            summary["top_clusters"] = [
                c for c in summary["top_clusters"] 
                if str(c.get("status")).lower() == "promoted"
            ]
                        
        return cls(config, summary, encoder)

    def get_support(self, query: ReviewExample, aspect: str) -> float:
        top_clusters = self.memory_summary.get("top_clusters", [])
        
        query_text = query.text.lower()
        query_emb = None
        if self.encoder is not None:
            if query_text not in self.query_cache:
                with torch.no_grad():
                    self.query_cache[query_text] = self.encoder.encode([query_text])
            query_emb = self.query_cache[query_text]
        
        best = 0.0
        
        for i, entry in enumerate(top_clusters):
            # Final safety check: status must be promoted
            if str(entry.get("status")).lower() != "promoted":
                continue

            canonical = entry.get("suggested_aspect") or entry.get("aspect_raw")
            if canonical != aspect:
                continue

            trigger_patterns = entry.get("trigger_patterns", [])
            representative = entry.get("representative_trigger")
            if representative and representative not in trigger_patterns:
                trigger_patterns.append(representative)

            sim = 0.0
            if self.encoder is not None and query_emb is not None and i in self.trigger_embs_cache:
                with torch.no_grad():
                    trigger_embs = self.trigger_embs_cache[i]
                    sims = F.cosine_similarity(query_emb, trigger_embs)
                    sim = sims.max().item()
                    # Apply a threshold to zero out weak semantic matches
                    if sim < self.config.memory_sim_threshold:
                        sim = 0.0
            else:
                for pattern in trigger_patterns:
                    if pattern.lower() in query_text:
                        sim = max(sim, 0.9)

            consistency = entry.get("consistency", entry.get("cluster_consistency", 0.75))
            quality = entry.get("quality", entry.get("evidence_quality_mean", 0.75))

            best = max(best, sim * consistency * quality)
            
        return best
