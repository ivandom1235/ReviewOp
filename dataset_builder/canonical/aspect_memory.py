from __future__ import annotations
import json
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Dict, List

from .evidence_clusterer import EvidenceClusterer
from .cluster_validator import ClusterValidator
from .cluster_labeler import ClusterLabeler
from .learned_hint_store import LearnedHintStore

@dataclass
class MemoryEntry:
    cluster_id: str
    aspect_raw: str  # Representative raw aspect
    status: str = "detected"
    support_count: int = 0
    unique_reviews: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    trigger_patterns: list[str] = field(default_factory=list)
    evidence_examples: list[dict[str, Any]] = field(default_factory=list)
    
    # Validation Metrics
    cluster_consistency: float = 0.0
    evidence_quality_mean: float = 0.0
    contradiction_score: float = 0.0
    
    # Labels
    suggested_aspect: Optional[str] = None
    generic_parent: Optional[str] = None
    generic_parent_status: str = "not_assigned"
    
    # Metadata
    validation_status: str = "unverified"
    run_id: str = ""
    last_used_at: Optional[str] = None
    
    @property
    def unique_review_count(self) -> int:
        return len(self.unique_reviews)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["unique_reviews"] = list(self.unique_reviews)
        d["domains"] = list(self.domains)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryEntry:
        # Legacy support: handle old entries without cluster_id
        if "cluster_id" not in d:
            raw = str(d.get("aspect_raw", "") or "unknown")
            d["cluster_id"] = f"legacy_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"
        
        # Legacy support: convert string evidence to dicts
        if "evidence_examples" in d:
            new_ev = []
            for ev in d["evidence_examples"]:
                if isinstance(ev, str):
                    new_ev.append({
                        "evidence_text": ev,
                        "review_id": "legacy",
                        "domain": "unknown",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    new_ev.append(ev)
            d["evidence_examples"] = new_ev

        d["unique_reviews"] = set(d.get("unique_reviews", []))
        d["domains"] = set(d.get("domains", []))
        return cls(**d)


class AspectMemory:
    HARD_REJECTS = {"unknown", "none", "null", "general", "misc", "something", "everything", "thing", "item"}
    SENTIMENT_CUES = {"good", "bad", "great", "poor", "nice", "awful", "excellent", "terrible", "best", "worst", "love", "hate", "amazing", "horrible", "friendly"}
    BEHAVIOR_CUES = {"broke", "broken", "fraying", "loose", "dropped", "dropping", "crashed", "waited", "waiting", "slow", "fast", "cold", "hot", "tiny", "small", "expensive", "cheap", "stale", "late", "delayed", "disconnected", "logged", "logging", "logout", "noisy", "buffering", "weak", "soggy", "undercooked", "responsive", "frayed", "opened", "fray", "frays", "cut", "cutting", "died"}
    GENERIC_EVIDENCE_PHRASES = {"great evening", "good evening", "great experience", "overall experience", "the restaurant", "the place", "the product", "the item", "the thing"}

    def __init__(
        self,
        storage_path: str | Path,
        auto_promote: bool = False,
        hint_store_path: Optional[str | Path] = None
    ):
        self.storage_path = Path(storage_path)
        self.auto_promote = auto_promote
        self.entries: dict[str, MemoryEntry] = {}
        
        self.clusterer = EvidenceClusterer(threshold=0.75)
        self.validator = ClusterValidator()
        self.labeler = ClusterLabeler()
        
        hint_path = hint_store_path or self.storage_path.parent / "aspect_hint_store.json"
        self.hint_store = LearnedHintStore(hint_path)
        
        self.load()

    def find_cluster_for_trigger(self, trigger: str) -> Optional[MemoryEntry]:
        """Finds an existing cluster that matches the given trigger pattern."""
        trigger_norm = self._normalize_pattern(trigger)
        if not self.entries or not trigger_norm:
            return None
            
        # 1. Exact match in trigger patterns
        for entry in self.entries.values():
            if trigger_norm in [self._normalize_pattern(p) for p in entry.trigger_patterns]:
                return entry
                
        # 2. Semantic match using clusterer
        clusters_list = [
            {
                "cluster_id": e.cluster_id,
                "representative_pattern": e.trigger_patterns[0] if e.trigger_patterns else e.aspect_raw,
                "trigger_patterns": list(e.trigger_patterns),
            }
            for e in self.entries.values() if e.trigger_patterns or e.aspect_raw
        ]
        
        best = self.clusterer.find_best_cluster(trigger_norm, clusters_list)
        if best:
            return self.entries.get(best["cluster_id"])
            
        return None

    def get_entry(self, key: str) -> Optional[MemoryEntry]:
        """Backward compatibility for tests and legacy callers."""
        key_norm = key.lower().strip()
        for entry in self.entries.values():
            if entry.cluster_id == key_norm:
                return entry
            if entry.aspect_raw.lower().strip() == key_norm:
                return entry
            if entry.suggested_aspect and entry.suggested_aspect.lower().strip() == key_norm:
                return entry
            if key_norm in [p.lower().strip() for p in entry.trigger_patterns]:
                return entry
        return None

    def add_evidence(
        self,
        aspect_raw: str,
        review_id: str,
        evidence_text: str,
        domain: str,
        *,
        sentiment: str = "unknown",
        run_id: Optional[str] = None,
    ) -> str:
        """
        Main entry point for adding evidence. Returns the cluster_id.
        """
        raw_val = aspect_raw.lower().strip()
        
        # Phase 3: Hard Rejects
        if raw_val in self.HARD_REJECTS or len(raw_val) < 2:
            return "rejected_noise"

        # Step 1: Extract trigger pattern
        trigger = self._extract_trigger_pattern(aspect_raw, evidence_text)
        if not trigger:
            return "rejected_noise"
        
        # Step 2: Find or Create Cluster
        entry = self.find_cluster_for_trigger(trigger)
        if not entry:
            cluster_id = f"mem_{hashlib.sha1(trigger.encode('utf-8')).hexdigest()[:8]}"
            entry = MemoryEntry(cluster_id=cluster_id, aspect_raw=aspect_raw)
            self.entries[cluster_id] = entry
            
        # Step 3: Update Entry
        entry.support_count += 1
        entry.unique_reviews.add(review_id)
        entry.domains.add(domain)
        
        if trigger not in entry.trigger_patterns:
            entry.trigger_patterns.append(trigger)
            
        if len(entry.evidence_examples) < 10:
            entry.evidence_examples.append({
                "review_id": review_id,
                "domain": domain,
                "evidence_text": evidence_text,
                "sentiment": sentiment,
                "trigger_pattern": trigger,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        entry.last_used_at = datetime.now(timezone.utc).isoformat()
        if not entry.run_id:
            entry.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            
        # Step 4: Refresh Metrics and Status
        self._refresh_cluster_metrics(entry)
        self._update_status(entry)
        
        return entry.cluster_id

    def _extract_trigger_pattern(self, aspect_raw: str, text: str) -> str:
        text = self._normalize_pattern(text)
        if not text:
            return ""

        aspect_norm = self._normalize_pattern(aspect_raw)
        clauses = self._split_clauses(text)
        aspect_clause = self._best_clause_for_aspect(aspect_norm, clauses)
        if aspect_clause:
            compact = self._compact_phrase(aspect_clause, aspect_norm)
            if compact and not self._is_weak_sentiment_phrase(compact):
                return compact

        behavior_clause = self._best_behavior_clause(clauses)
        if behavior_clause:
            compact = self._compact_phrase(behavior_clause, aspect_norm)
            if compact and not self._is_weak_sentiment_phrase(compact):
                return compact

        return ""

    def _refresh_cluster_metrics(self, entry: MemoryEntry):
        # Calculate real cluster consistency
        unique_surface_forms = self._unique_surface_form_count(entry.trigger_patterns)
        if len(entry.trigger_patterns) < 2:
            entry.cluster_consistency = 1.0 if entry.support_count >= 2 and unique_surface_forms >= 2 else 0.0
        else:
            entry.cluster_consistency = self._mean_pairwise_similarity(entry.trigger_patterns)

        # Calculate evidence quality
        if entry.evidence_examples:
            qualities = [self._basic_evidence_quality(e["evidence_text"]) for e in entry.evidence_examples]
            entry.evidence_quality_mean = sum(qualities) / len(qualities)
        else:
            entry.evidence_quality_mean = 0.0
            
        # Labeling (Phase 2)
        if not entry.suggested_aspect or entry.support_count < 5 or entry.support_count % 5 == 0:
            entry.suggested_aspect = self.labeler.label_cluster(entry.trigger_patterns, entry.aspect_raw)
            entry.generic_parent = None
            entry.generic_parent_status = "not_assigned"

    def _mean_pairwise_similarity(self, patterns: list[str]) -> float:
        return self.clusterer.mean_similarity(patterns)

    def _basic_evidence_quality(self, text: str) -> float:
        text = str(text or "").strip()
        if not text: return 0.0
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return 0.0

        token_set = set(tokens)
        score = 0.30

        if 3 <= len(tokens) <= 20:
            score += 0.20
        if token_set & self.BEHAVIOR_CUES:
            score += 0.25
        if token_set & self.SENTIMENT_CUES and len(tokens) > 1:
            score += 0.15
        if len(tokens) > 40:
            score -= 0.30
        if len(tokens) <= 1:
            score -= 0.30
        if self._is_generic_evidence_phrase(text):
            score -= 0.20
        if self._looks_entity_only_phrase(tokens):
            score -= 0.30

        return max(0.0, min(1.0, score))

    def _update_status(self, entry: MemoryEntry):
        if entry.status in {"promoted", "rejected"}:
            return

        metrics = {
            "support_count": entry.support_count,
            "unique_review_count": entry.unique_review_count,
            "cluster_consistency": entry.cluster_consistency,
            "evidence_quality_mean": entry.evidence_quality_mean,
            "contradiction_score": entry.contradiction_score,
            "aspect_raw": entry.aspect_raw,
            "trigger_patterns": tuple(entry.trigger_patterns),
            "unique_surface_form_count": self._unique_surface_form_count(entry.trigger_patterns),
        }
        # print(f"DEBUG AspectMemory: {entry.cluster_id} - {metrics}")
        
        if self.validator.validate_for_review_queue(metrics):
            entry.status = "review_queue"
            entry.validation_status = "unverified"
            
            # Check LearnedHintStore for pre-approved clusters
            hint = self.hint_store.get_cluster(entry.cluster_id)
            if hint and hint.get("status") == "promoted":
                entry.status = "promoted"
                entry.validation_status = "learned_promoted"
                if hint.get("suggested_aspect"):
                    entry.suggested_aspect = hint.get("suggested_aspect")
            
            # Auto-promote logic
            if entry.status != "promoted" and self.auto_promote and entry.support_count >= 10:
                entry.status = "promoted"
                entry.validation_status = "auto_validated"
        else:
            entry.status = "detected"

    def write_summary(self, output_path: str | Path) -> None:
        total = len(self.entries)
        review_queue = [e for e in self.entries.values() if e.status == "review_queue"]
        promoted = [e for e in self.entries.values() if e.status == "promoted"]
        bootstrap_entries = [e for e in self.entries.values() if self._is_bootstrap_entry(e)]
        organic_entries = [e for e in self.entries.values() if not self._is_bootstrap_entry(e)]
        
        # Calculate broad noun rate for summary
        BROAD_NOUNS = {
            "pizza", "restaurant", "computer", "windows", "table", "glass",
            "wine", "bread", "dinner", "lunch", "people", "lcd", "food",
            "staff", "place", "product", "item"
        }
        
        broad_count = 0
        unknown_count = 0
        for e in self.entries.values():
            raw_low = e.aspect_raw.lower()
            if raw_low == "unknown":
                unknown_count += 1
            candidates = [raw_low]
            if e.suggested_aspect:
                candidates.append(str(e.suggested_aspect).lower())
            candidates.extend(str(p).lower() for p in e.trigger_patterns)
            if any(self._is_broad_candidate(candidate, BROAD_NOUNS) for candidate in candidates):
                if not any(self._contains_behavior_cue(pattern) for pattern in e.trigger_patterns):
                    broad_count += 1
        
        evidence_pattern_count = sum(
            1
            for e in self.entries.values()
            if any(self._contains_behavior_cue(pattern) for pattern in e.trigger_patterns)
        )

        payload = {
            "total_entries": total,
            "review_queue_count": len(review_queue),
            "promoted_count": len(promoted),
            "rejected_count": sum(1 for e in self.entries.values() if e.status == "rejected"),
            "bootstrap_entry_count": len(bootstrap_entries),
            "organic_entry_count": len(organic_entries),
            "bootstrap_review_queue_count": sum(1 for e in bootstrap_entries if e.status == "review_queue"),
            "organic_review_queue_count": sum(1 for e in organic_entries if e.status == "review_queue"),
            "unknown_candidate_count": unknown_count,
            "broad_noun_candidate_rate": broad_count / total if total > 0 else 0,
            "evidence_pattern_candidate_rate": evidence_pattern_count / total if total > 0 else 0,
            "top_clusters": [
                {
                    "cluster_id": e.cluster_id,
                    "suggested_aspect": e.suggested_aspect,
                    "trigger_patterns": e.trigger_patterns[:3],
                    "support_count": e.support_count,
                    "status": e.status,
                    "consistency": e.cluster_consistency,
                    "quality": e.evidence_quality_mean
                }
                for e in sorted(self.entries.values(), key=lambda x: x.support_count, reverse=True)[:20]
            ],
            "top_candidates": [
                {
                    "cluster_id": e.cluster_id,
                    "suggested_aspect": e.suggested_aspect,
                    "trigger_patterns": e.trigger_patterns[:3],
                    "support_count": e.support_count,
                    "status": e.status,
                    "consistency": e.cluster_consistency,
                    "quality": e.evidence_quality_mean
                }
                for e in sorted(self.entries.values(), key=lambda x: x.support_count, reverse=True)[:20]
            ],
        }
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _is_bootstrap_entry(entry: MemoryEntry) -> bool:
        if str(entry.run_id or "").strip().lower() == "synthetic_bootstrap":
            return True
        return bool(entry.evidence_examples) and all(
            str(example.get("timestamp", "") or "").strip().lower() == "synthetic"
            for example in entry.evidence_examples
        )

    def match_promoted(self, text: str) -> list[MemoryEntry]:
        """Finds all promoted clusters that have a presence in the given text."""
        text_norm = self._normalize_pattern(text)
        if not text_norm:
            return []
        matches: list[MemoryEntry] = []
        segments = self._text_segments(text_norm)
        for entry in self.entries.values():
            if entry.status != "promoted":
                continue
            # Match against suggested_aspect or any known trigger pattern
            patterns = [p for p in ([entry.suggested_aspect] if entry.suggested_aspect else []) + list(entry.trigger_patterns) if p]

            for p in patterns:
                p_norm = self._normalize_pattern(p)
                if not p_norm:
                    continue
                if self._phrase_in_text(p_norm, text_norm):
                    entry.last_used_at = datetime.now(timezone.utc).isoformat()
                    matches.append(entry)
                    break
                if any(self.clusterer.get_similarity_score(p_norm, segment) >= 0.78 for segment in segments):
                    entry.last_used_at = datetime.now(timezone.utc).isoformat()
                    matches.append(entry)
                    break
        return matches

    def write_review_queue(self, output_path: str | Path) -> None:
        """Exports clusters in 'review_queue' status for manual validation."""
        items = []
        for e in self.entries.values():
            if e.status != "review_queue":
                continue
            items.append({
                "cluster_id": e.cluster_id,
                "aspect_raw": e.aspect_raw,
                "suggested_aspect": e.suggested_aspect,
                "support_count": e.support_count,
                "unique_review_count": e.unique_review_count,
                "trigger_patterns": e.trigger_patterns,
                "evidence_examples": e.evidence_examples[:3],
                "consistency": e.cluster_consistency,
                "quality": e.evidence_quality_mean
            })
        payload = {"created_at": datetime.now(timezone.utc).isoformat(), "total": len(items), "items": items}
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save(self):
        """Persists the memory to the storage path and syncs hint store."""
        # Sync promoted clusters to hint store
        for e in self.entries.values():
            if e.status == "promoted":
                self.hint_store.add_cluster(e.cluster_id, e.to_dict())
        
        payload = {
            "entries": {k: e.to_dict() for k, e in self.entries.items()},
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self):
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.entries = {k: MemoryEntry.from_dict(v) for k, v in data.get("entries", {}).items()}
        except Exception:
            self.entries = {}

    @staticmethod
    def _normalize_pattern(text: str) -> str:
        return " ".join(str(text or "").lower().split())

    @staticmethod
    def _surface_form(text: str) -> str:
        return " ".join(re.findall(r"\b\w+\b", str(text or "").lower()))

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        normalized = " ".join(str(text or "").split())
        parts = []
        for chunk in re.split(r"[.;,!?]|(?:\bbut\b|\band\b|\bhowever\b|\bthough\b|\byet\b)", normalized, flags=re.IGNORECASE):
            chunk = " ".join(chunk.split()).strip()
            if chunk:
                parts.append(chunk)
        return parts or ([normalized] if normalized else [])

    @staticmethod
    def _best_clause_for_aspect(aspect_norm: str, clauses: list[str]) -> str:
        if not aspect_norm:
            return ""
        for clause in clauses:
            if aspect_norm in clause.lower():
                return clause
        return ""

    @staticmethod
    def _best_behavior_clause(clauses: list[str]) -> str:
        cues = AspectMemory.BEHAVIOR_CUES
        sentiment_cues = AspectMemory.SENTIMENT_CUES
        best_clause = ""
        best_score = 0
        for clause in clauses:
            tokens = clause.lower().split()
            cue_hits = sum(1 for token in tokens if token.strip(".,!?") in cues)
            if cue_hits == 0:
                sentiment_hits = sum(1 for token in tokens if token.strip(".,!?") in sentiment_cues)
                if sentiment_hits == 0:
                    continue
                if len(tokens) <= 2 or AspectMemory._looks_generic_clause(tokens):
                    continue
                cue_hits = sentiment_hits
            length = len(tokens)
            if 3 <= length <= 12:
                score = cue_hits * 3 + (12 - abs(7 - length))
            else:
                score = cue_hits * 2
            if score > best_score:
                best_score = score
                best_clause = clause
        return best_clause

    @staticmethod
    def _compact_phrase(text: str, aspect_norm: str) -> str:
        cleaned = " ".join(str(text or "").split()).strip(" ,.;:!?")
        if not cleaned:
            return ""
        original = cleaned.lower()
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8])
        cleaned = AspectMemory._trim_weak_start_tokens(cleaned)
        cleaned_tokens = re.findall(r"\b\w+\b", cleaned.lower())
        if cleaned.lower() != original:
            if not cleaned_tokens or not any(token in AspectMemory.BEHAVIOR_CUES for token in cleaned_tokens):
                return ""
        if aspect_norm and cleaned.lower() == aspect_norm:
            return ""
        return cleaned.lower()

    @staticmethod
    def _phrase_in_text(pattern: str, text_norm: str) -> bool:
        return bool(re.search(rf"\b{re.escape(pattern)}\b", text_norm))

    @staticmethod
    def _text_segments(text: str) -> list[str]:
        pieces = []
        for chunk in re.split(r"[.;!?]", text):
            chunk = " ".join(chunk.split()).strip()
            if chunk:
                pieces.append(chunk)
        return pieces or ([text] if text else [])

    def _unique_surface_form_count(self, trigger_patterns: list[str]) -> int:
        return len({self._surface_form(pattern) for pattern in trigger_patterns if self._surface_form(pattern)})

    def _contains_behavior_cue(self, pattern: str) -> bool:
        tokens = set(re.findall(r"\b\w+\b", str(pattern or "").lower()))
        return bool(tokens & self.BEHAVIOR_CUES)

    def _is_generic_evidence_phrase(self, text: str) -> bool:
        normalized = self._normalize_pattern(text)
        return normalized in self.GENERIC_EVIDENCE_PHRASES

    def _looks_entity_only_phrase(self, tokens: list[str]) -> bool:
        return len(tokens) == 1 or (len(tokens) == 2 and all(len(token) <= 4 for token in tokens))

    def _is_broad_candidate(self, candidate: str, broad_nouns: set[str]) -> bool:
        tokens = self._surface_form(candidate).split()
        if not tokens:
            return False
        if any(token in self.BEHAVIOR_CUES for token in tokens):
            return False
        return len(tokens) == 1 and tokens[0] in broad_nouns

    @staticmethod
    def _looks_generic_clause(tokens: list[str]) -> bool:
        generic = {"evening", "experience", "overall", "place", "thing", "stuff", "product", "item", "service", "food", "staff", "restaurant"}
        return any(token.strip(".,!?") in generic for token in tokens)

    def _is_weak_sentiment_phrase(self, phrase: str) -> bool:
        tokens = re.findall(r"\b\w+\b", str(phrase or "").lower())
        if not tokens:
            return True
        token_set = set(tokens)
        if token_set & self.BEHAVIOR_CUES:
            return False
        if token_set & self.SENTIMENT_CUES:
            return True
        if self._is_generic_evidence_phrase(phrase):
            return True
        if len(tokens) == 2 and self._looks_generic_clause(tokens):
            return True
        return False

    @staticmethod
    def _trim_weak_start_tokens(text: str) -> str:
        tokens = re.findall(r"\b\w+\b", str(text or "").lower())
        weak_single_starts = {
            "that",
            "which",
            "where",
            "quality",
            "great",
            "excellent",
            "good",
            "bad",
            "poor",
            "awful",
            "terrible",
        }
        weak_start_pairs = {
            ("i", "was"),
            ("i", "am"),
            ("they", "have"),
            ("we", "have"),
            ("it", "was"),
            ("it", "is"),
            ("that", "was"),
            ("that", "is"),
            ("this", "was"),
            ("this", "is"),
        }
        while tokens:
            if len(tokens) >= 2 and (tokens[0], tokens[1]) in weak_start_pairs:
                tokens = tokens[2:]
                continue
            if tokens[0] in weak_single_starts:
                tokens = tokens[1:]
                continue
            break
        return " ".join(tokens).strip()
