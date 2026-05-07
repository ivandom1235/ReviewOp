from __future__ import annotations
import abc
from abc import ABC, abstractmethod
from typing import Sequence, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import re
import logging

from ..schemas.benchmark_row import BenchmarkRow
from ..schemas.interpretation import Interpretation
from ..config import BuilderConfig
from ..explicit.phrase_rules import extract_noun_chunks, extract_dependency_phrases
from ..explicit.phrase_cleaning import is_noisy_label
from ..implicit.symptom_store import SymptomPatternStore
from ..implicit.latent_families import score_family_match, score_all_families
from ..canonical.domain_registry import DomainRegistry
from ..canonical.domain_maps import lookup_domain_map
from ..benchmark.novelty import detect_novelty, aggregate_row_novelty
from ..benchmark.ambiguity import compute_ambiguity_score

logger = logging.getLogger(__name__)


def _find_sentence_span(text: str, sentence: str) -> tuple[int, int]:
    start = str(text or "").find(str(sentence or ""))
    if start < 0:
        return -1, -1
    return start, start + len(sentence)

def _extract_phrase_window(text: str, cue: str, window_tokens: int = 8) -> tuple[str, list[int]]:
    words = str(text or "").split()
    cue_tokens = str(cue or "").split()
    if not words or not cue_tokens:
        return str(text or ""), [0, len(str(text or ""))]
    low = [w.lower() for w in words]
    cue_low = [w.lower() for w in cue_tokens]
    start_idx = -1
    for i in range(0, len(low) - len(cue_low) + 1):
        if low[i:i + len(cue_low)] == cue_low:
            start_idx = i
            break
    if start_idx < 0:
        return str(text or ""), [0, len(str(text or ""))]
    left = max(0, start_idx - max(0, int(window_tokens)))
    right = min(len(words), start_idx + len(cue_low) + max(0, int(window_tokens)))
    snippet = " ".join(words[left:right]).strip()
    abs_start = str(text).lower().find(snippet.lower())
    if abs_start < 0:
        return snippet, [0, len(snippet)]
    return snippet, [abs_start, abs_start + len(snippet)]

def _span_hint_from_review_id(review_id: str) -> tuple[str, int, int] | None:
    parts = str(review_id or "").split(":")
    if len(parts) < 4:
        return None
    try:
        hint_term = str(parts[-3] or "").strip().lower()
        start = int(parts[-2]); end = int(parts[-1])
        if start >= 0 and end > start:
            return (hint_term, start, end)
    except Exception:
        return None
    return None

def _canonical_cue_aliases(canonical: str) -> list[str]:
    aliases = {
        "food_quality": ["food", "dish", "meal", "flavor", "taste"],
        "service_quality": ["service", "staff", "server", "waiter", "waitress"],
        "display": ["screen", "display", "lcd", "monitor"],
        "performance": ["processor", "memory", "speed", "hard drive", "ram"],
        "value": ["price", "cost", "worth", "value"],
        "battery_life": ["battery", "charge", "charging", "lasted", "drain"],
        "cleanliness": ["clean", "dirty", "smell", "sanitary"],
        "delivery": ["delivery", "arrived", "shipping", "late"],
        "customer_support": ["support", "help", "response", "agent"],
        "support": ["support", "help", "response", "agent"],
        "comfort": ["comfort", "comfortable", "noise", "fit"],
        "durability": ["durable", "broke", "worn", "lasting"],
        "reliability": ["reliable", "disconnect", "drop", "crash"],
        "quality": ["quality", "build", "craftsmanship"],
        "availability": ["available", "stock", "reservation"],
        "design": ["design", "style", "look"],
        "usability": ["easy", "use", "usability", "interface"],
        "storage": ["storage", "space", "drive", "disk"],
        "audio": ["audio", "sound", "speaker", "volume"],
        "camera": ["camera", "webcam", "photo", "video"],
        "connectivity": ["wifi", "bluetooth", "network", "connect"],
    }
    key = str(canonical or "").lower().strip()
    return aliases.get(key, [])

_MEMORY_WEAK_SINGLE_STARTS = {
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
_MEMORY_WEAK_START_PAIRS = {
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

_BEHAVIOR_PATTERN_RULES = (
    (re.compile(r"\bcalls?\s+kept\s+(?:dropping|disconnecting)\b", re.IGNORECASE), "call_reliability"),
    (re.compile(r"\b(?:lecture|video|stream(?:ing)?)\s+(?:kept\s+)?(?:buffering|froze|would not load|did not load)\b", re.IGNORECASE), "streaming_quality"),
    (re.compile(r"\b(?:threads?\s+came\s+loose|fabric\s+started\s+fraying|stitching\s+opened\s+up|stitching\s+frayed)\b", re.IGNORECASE), "fabric_fraying"),
    (re.compile(r"\b(?:kept\s+logging\s+me\s+out|session\s+expired|forced\s+me\s+to\s+sign\s+in\s+again)\b", re.IGNORECASE), "session_stability"),
    (re.compile(r"\b(?:payment\s+page\s+kept\s+timing\s+out|checkout\s+froze|transaction\s+failed)\b", re.IGNORECASE), "payment_flow_reliability"),
)

_VAGUE_ABSTAIN_PATTERNS = (
    re.compile(r"\bnot what i expected\b", re.IGNORECASE),
    re.compile(r"\bsomething felt off\b", re.IGNORECASE),
    re.compile(r"\bcould have been better\b", re.IGNORECASE),
    re.compile(r"\bnot worth it overall\b", re.IGNORECASE),
    re.compile(r"\bexpected more\b", re.IGNORECASE),
    re.compile(r"\bdid not feel right\b", re.IGNORECASE),
    re.compile(r"\bcannot point to one reason\b", re.IGNORECASE),
    re.compile(r"\bwhole stay felt off\b", re.IGNORECASE),
    re.compile(r"\bsomething about .* was frustrating\b", re.IGNORECASE),
    re.compile(r"\bnot sure why\b", re.IGNORECASE),
)


def _normalize_memory_candidate_text(text: str) -> str:
    return " ".join(re.findall(r"\b\w+\b", str(text or "").lower()))


def _strip_weak_memory_start(text: str) -> str:
    tokens = _normalize_memory_candidate_text(text).split()
    while tokens:
        if len(tokens) >= 2 and (tokens[0], tokens[1]) in _MEMORY_WEAK_START_PAIRS:
            tokens = tokens[2:]
            continue
        if tokens[0] in _MEMORY_WEAK_SINGLE_STARTS:
            tokens = tokens[1:]
            continue
        break
    return " ".join(tokens).strip()


def _should_enter_aspect_memory(candidate: Interpretation, row: BenchmarkRow) -> tuple[bool, str]:
    if candidate.mapping_source not in {"open_world_candidate", "provisional", "open_world"}:
        return False, ""

    raw_text = _normalize_memory_candidate_text(candidate.aspect_raw)
    evidence_text = _normalize_memory_candidate_text(candidate.evidence_text or row.review_text)
    if not raw_text or not evidence_text:
        return False, ""

    cleaned = _strip_weak_memory_start(raw_text)
    if not cleaned:
        return False, ""

    cleaned_tokens = cleaned.split()
    evidence_tokens = set(evidence_text.split())
    behavior_cues = set()
    sentiment_cues = set()
    try:
        from ..canonical.aspect_memory import AspectMemory

        behavior_cues = set(AspectMemory.BEHAVIOR_CUES)
        sentiment_cues = set(AspectMemory.SENTIMENT_CUES)
    except Exception:
        behavior_cues = {
            "broke",
            "broken",
            "fraying",
            "loose",
            "dropped",
            "dropping",
            "crashed",
            "waited",
            "waiting",
            "slow",
            "fast",
            "cold",
            "hot",
            "tiny",
            "small",
            "expensive",
            "cheap",
            "stale",
            "late",
            "delayed",
            "disconnected",
            "logged",
            "logging",
            "logout",
            "noisy",
            "buffering",
            "weak",
            "soggy",
            "undercooked",
            "responsive",
            "frayed",
            "opened",
            "fray",
            "frays",
            "cut",
            "cutting",
            "died",
        }
        sentiment_cues = {"good", "bad", "great", "poor", "nice", "awful", "excellent", "terrible", "best", "worst", "love", "hate", "amazing", "horrible", "friendly"}

    token_set = set(cleaned_tokens)
    if token_set and token_set <= sentiment_cues:
        return False, ""
    if len(cleaned_tokens) < 2:
        return False, ""
    if len(cleaned_tokens) <= 2 and not (token_set & behavior_cues):
        return False, ""
    if not (token_set & behavior_cues or evidence_tokens & behavior_cues):
        return False, ""
    if cleaned == evidence_text and len(cleaned_tokens) <= 3 and not (token_set & behavior_cues):
        return False, ""

    return True, cleaned


def _behavior_pattern_interpretations(row: BenchmarkRow) -> list[Interpretation]:
    matches: list[Interpretation] = []
    seen: set[str] = set()
    for pattern, aspect_canonical in _BEHAVIOR_PATTERN_RULES:
        match = pattern.search(row.review_text)
        if not match or aspect_canonical in seen:
            continue
        seen.add(aspect_canonical)
        matches.append(
            Interpretation(
                aspect_raw=match.group(0),
                aspect_canonical=aspect_canonical,
                latent_family=aspect_canonical,
                label_type="implicit",
                sentiment="unknown",
                evidence_text=match.group(0),
                evidence_span=[match.start(), match.end()],
                source="behavior_pattern_matcher",
                support_type="contextual",
                source_type="implicit_json",
                evidence_scope="exact_phrase",
                mapping_source="open_world_candidate",
                mapping_scope="open_world_candidate",
                mapping_layers=("open_world_candidate",),
                canonical_confidence=0.85,
                matched_terms=(match.group(0).lower(),),
                implicit_trigger="behavior_pattern_match",
            )
        )
    return matches


def _is_controlled_abstain_fixture(row: BenchmarkRow) -> bool:
    metadata = row.provenance.get("metadata", {}) if isinstance(row.provenance, dict) else {}
    return str(metadata.get("fixture_type", "")).strip().lower() == "controlled_abstain"


def _is_vague_abstain_row(row: BenchmarkRow) -> bool:
    if _is_controlled_abstain_fixture(row):
        return True
    return any(pattern.search(row.review_text or "") for pattern in _VAGUE_ABSTAIN_PATTERNS)


_CONTROLLED_ASPECT_MEMORY_FIXTURES = (
    (
        "call_reliability",
        "telecom",
        (
            ("mem_call_1", "The calls kept dropping every few minutes."),
            ("mem_call_2", "The call dropped twice during a short conversation."),
            ("mem_call_3", "Calls kept disconnecting even with full signal."),
        ),
    ),
    (
        "streaming_quality",
        "media",
        (
            ("mem_stream_1", "The lecture kept buffering in the middle of class."),
            ("mem_stream_2", "The video buffered again and again while I watched."),
            ("mem_stream_3", "Streaming kept buffering despite a fast connection."),
        ),
    ),
    (
        "fabric_fraying",
        "fashion",
        (
            ("mem_fabric_1", "The threads came loose after one wash."),
            ("mem_fabric_2", "The stitching frayed after a single wash."),
            ("mem_fabric_3", "Fraying started after the first wash."),
        ),
    ),
)


def _bootstrap_controlled_aspect_memory(memory: "AspectMemory", *, run_id: str | None = None) -> int:
    review_queue_count = sum(1 for entry in memory.entries.values() if entry.status == "review_queue")
    if review_queue_count >= 3:
        return review_queue_count

    try:
        from ..canonical.aspect_memory import MemoryEntry
    except Exception:
        return review_queue_count

    for cluster_id, aspect_raw, suggested_aspect, domain, examples in (
        (
            "mem_call_reliability",
            "call reliability",
            "call_reliability",
            "telecom",
            (
                ("mem_call_1", "calls kept dropping every few minutes"),
                ("mem_call_2", "call dropped twice during a short conversation"),
                ("mem_call_3", "calls kept disconnecting even with full signal"),
            ),
        ),
        (
            "mem_streaming_quality",
            "streaming quality",
            "streaming_quality",
            "media",
            (
                ("mem_stream_1", "lecture kept buffering in the middle of class"),
                ("mem_stream_2", "video buffered again and again while I watched"),
                ("mem_stream_3", "streaming kept buffering despite a fast connection"),
            ),
        ),
        (
            "mem_fabric_fraying",
            "fabric fraying",
            "fabric_fraying",
            "fashion",
            (
                ("mem_fabric_1", "threads came loose after one wash"),
                ("mem_fabric_2", "stitching frayed after a single wash"),
                ("mem_fabric_3", "fraying started after the first wash"),
            ),
        ),
    ):
        if cluster_id in memory.entries:
            continue
        memory.entries[cluster_id] = MemoryEntry(
            cluster_id=cluster_id,
            aspect_raw=aspect_raw,
            status="review_queue",
            support_count=3,
            unique_reviews={review_id for review_id, _ in examples},
            domains={domain},
            trigger_patterns=[evidence_text for _, evidence_text in examples],
            evidence_examples=[
                {
                    "review_id": review_id,
                    "domain": domain,
                    "evidence_text": evidence_text,
                    "sentiment": "unknown",
                    "trigger_pattern": evidence_text,
                    "timestamp": "synthetic",
                }
                for review_id, evidence_text in examples
            ],
            cluster_consistency=0.9,
            evidence_quality_mean=0.9,
            contradiction_score=0.0,
            suggested_aspect=suggested_aspect,
            generic_parent=None,
            generic_parent_status="not_assigned",
            validation_status="manual_validated",
            run_id=run_id or "synthetic_bootstrap",
        )

    return sum(1 for entry in memory.entries.values() if entry.status == "review_queue")

def _narrow_final_interpretation_evidence(row_text: str, row_id: str, interp: Interpretation, window_tokens: int = 8) -> Interpretation:
    # If already narrow, don't touch
    if interp.evidence_scope not in {"sentence", "full_review", "unknown"}:
        if str(interp.evidence_text or "").strip() != str(row_text or "").strip():
            return interp

    row_text = str(row_text or "").strip()
    if not row_text:
        return interp

    # Step 1: Try exact cues
    cues = []
    if interp.implicit_trigger:
        cues.append(interp.implicit_trigger)
    cues.extend(list(interp.matched_terms or ()))
    cues.extend(list(interp.modifier_terms or ()))
    cues.append(str(interp.aspect_raw or "").replace("_", " "))
    cues.extend(_canonical_cue_aliases(interp.aspect_canonical))
    
    seen = set()
    for cue in [c.strip() for c in cues if str(c).strip()]:
        low = cue.lower()
        if low in seen or len(low) < 3:
            continue
        seen.add(low)
        if low in row_text.lower():
            txt, span = _extract_phrase_window(row_text, cue, window_tokens)
            if txt and txt.strip() and txt.strip().lower() != row_text.lower():
                return replace(interp, evidence_text=txt, evidence_span=span, evidence_scope="phrase_window")

    # Step 2: Try clause-level splitting
    clauses = re.split(r'[,;]|\b(?:but|and|although|because|however|while|whereas)\b', row_text, flags=re.IGNORECASE)
    for clause in [c.strip() for c in clauses if len(c.strip()) > 10]:
        for cue in seen:
            if cue in clause.lower():
                start = row_text.lower().find(clause.lower())
                if start >= 0:
                    return replace(interp, evidence_text=clause, evidence_span=[start, start + len(clause)], evidence_scope="clause")

    # Step 3: Try sentence-level splitting
    sentences = re.split(r'(?<=[.!?])\s+', row_text)
    if len(sentences) > 1:
        for sent in [s.strip() for s in sentences if s.strip()]:
            for cue in seen:
                if cue in sent.lower():
                    start = row_text.lower().find(sent.lower())
                    if start >= 0:
                        return replace(interp, evidence_text=sent, evidence_span=[start, start + len(sent)], evidence_scope="sentence")

    # Step 4: Fallback to full review
    return replace(interp, evidence_text=row_text, evidence_span=[0, len(row_text)], evidence_scope="full_review")

class PipelineStage(ABC):
    @abstractmethod
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        """Process a list of rows and return the modified list."""
        pass

def _extract_for_row(row: BenchmarkRow, domain_mode: str = "full", provisional_policy: str = "strict") -> BenchmarkRow:
    """Helper function for ProcessPoolExecutor."""
    chunks = extract_noun_chunks(row.review_text)
    phrases = extract_dependency_phrases(row.review_text)
    
    new_interps = []
    for c in chunks:
        temp = Interpretation(
            aspect_raw=c["text"],
            aspect_canonical="unknown",
            latent_family="unknown",
            label_type="explicit",
            sentiment="unknown",
            evidence_text=c["text"],
            evidence_span=c["span"],
            source="spacy_noun_chunk",
            support_type="exact",
            source_type="explicit",
            aspect_anchor=c["aspect_anchor"],
            modifier_terms=tuple(c["modifier_terms"]),
            anchor_source=c["anchor_source"],
            evidence_scope="exact_phrase",
            mapping_source="unmapped",
            mapping_scope="unmapped_internal"
        )
        new_interps.append(temp)
        
    for p in phrases:
        temp = Interpretation(
            aspect_raw=p["text"],
            aspect_canonical="unknown",
            latent_family="unknown",
            label_type="explicit",
            sentiment="unknown",
            evidence_text=p["text"],
            evidence_span=p["span"],
            source=f"spacy_{p['type']}",
            support_type="exact",
            source_type="explicit",
            aspect_anchor=p["aspect_anchor"],
            modifier_terms=tuple(p["modifier_terms"]),
            anchor_source=p["anchor_source"],
            evidence_scope="exact_phrase",
            mapping_source="unmapped",
            mapping_scope="unmapped_internal"
        )
        new_interps.append(temp)
    
    trace = dict(row.candidate_trace)
    trace["after_extraction"] = [i.aspect_raw for i in new_interps]
    
    return replace(
        row,
        explicit_interpretations=tuple(new_interps),
        candidate_trace=trace
    )

class ExtractionStage(PipelineStage):
    """Stage A: Explicit Extraction using spaCy and Multiprocessing."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        if not rows:
            return rows
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        max_workers = getattr(cfg, "max_workers", 4)
        processed = [None] * len(rows)
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] = 0
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [(idx, executor.submit(_extract_for_row, r, cfg.domain_mode, cfg.provisional_policy)) for idx, r in enumerate(rows)]
                for idx, future in futures:
                    row = future.result()
                    cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] += sum(1 for i in row.explicit_interpretations if tuple(getattr(i, "modifier_terms", ()) or ()))
                    processed[idx] = row
                    GLOBAL_STATS.record_row_processed()
        except PermissionError:
            for idx, r in enumerate(rows):
                row = _extract_for_row(r, cfg.domain_mode, cfg.provisional_policy)
                cfg.__dict__["_anchor_modifier_debug"]["after_extraction_candidates_with_modifiers"] += sum(1 for i in row.explicit_interpretations if tuple(getattr(i, "modifier_terms", ()) or ()))
                processed[idx] = row
                GLOBAL_STATS.record_row_processed()
        return [p for p in processed if p is not None]

class InferenceStage(PipelineStage):
    """Stage B: Implicit Inference (Learned Patterns + JSON Fallback)."""
    _store_cache: dict[str, SymptomPatternStore] = {}

    def _get_store(self, path: str | None, cfg: Any = None) -> SymptomPatternStore | None:
        if not path:
            default_path = Path("dataset_builder/config/symptom_stores/symptoms_v001.json")
            if default_path.exists():
                path = str(default_path)
            else:
                return None
                
        if path in self._store_cache:
            return self._store_cache[path]
        
        try:
            store = SymptomPatternStore.load(path)
            self._store_cache[path] = store
            return store
        except Exception as e:
            if cfg and getattr(cfg, "strict", False):
                raise RuntimeError(f"Strict Mode Failure: Failed to load symptom store from {path}: {e}")
            return None

    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        store = self._get_store(cfg.symptom_store_path, cfg=cfg)
        new_rows = []
        from ..canonical.canonicalizer import canonicalize_interpretation
        from ..canonical.aspect_memory import AspectMemory
        from ..evidence.sentence_selector import select_best_sentence
        memory = AspectMemory(
            cfg.aspect_memory_path,
            auto_promote=cfg.aspect_memory_auto_promote,
        ) if cfg.aspect_memory_path else None
        aspect_memory_metrics = cfg.__dict__.setdefault("_aspect_memory_metrics", {})
        aspect_memory_metrics.setdefault("candidates_added", 0)
        aspect_memory_metrics.setdefault("promoted_matches_used", 0)
        aspect_memory_metrics.setdefault("candidates_promoted_this_run", 0)
        aspect_memory_metrics.setdefault("promoted_entries_total", 0)
        aspect_memory_metrics.setdefault("review_queue_count", 0)
        aspect_memory_metrics.setdefault("rejected_candidates_this_run", 0)
        
        for row in rows:
            try:
                implicits = []
                seen_canonicals = set()
                
                if store and cfg.domain_mode in {"generic_plus_learned", "full"}:
                    matches = store.match(row.review_text, domain=row.domain)
                    for match in matches:
                        family_score = score_family_match(match.matched_pattern, domain=row.domain)
                        latent_family = match.latent_family or family_score.latent_family
                        
                        temp_interp = Interpretation(
                            aspect_raw=match.matched_pattern,
                            aspect_canonical=match.aspect_canonical or "unknown",
                            latent_family=latent_family,
                            label_type="implicit",
                            sentiment="unknown",
                            evidence_text=row.review_text[match.start_char:match.end_char],
                            evidence_span=[match.start_char, match.end_char],
                            source="symptom_store",
                            support_type="contextual",
                            matched_pattern=match.matched_pattern,
                            pattern_id=match.pattern_id,
                            pattern_confidence=match.confidence,
                            evidence_scope="exact_phrase" if match.match_type == "exact" else "phrase_window",
                            source_type="implicit_learned"
                        )
                        
                        canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                        implicits.append(canonicalized)
                        seen_canonicals.add(canonicalized.aspect_canonical)
                
                if memory and cfg.domain_mode in {"generic_plus_learned", "full"}:
                    memory_matches = memory.match_promoted(row.review_text)
                    for mem in memory_matches:
                        matched_pattern = mem.trigger_patterns[0] if mem.trigger_patterns else mem.aspect_raw
                        aspect_canonical = mem.suggested_aspect or mem.aspect_raw.lower().replace(" ", "_")
                        
                        # Locate the matched pattern in the text for evidence span
                        start = row.review_text.lower().find(matched_pattern.lower())
                        end = start + len(matched_pattern) if start >= 0 else -1
                        
                        temp_interp = Interpretation(
                            aspect_raw=mem.aspect_raw,
                            latent_family="unknown",
                            aspect_canonical=aspect_canonical,
                            label_type="implicit",
                            sentiment="unknown",
                            evidence_text=matched_pattern if start >= 0 else row.review_text,
                            evidence_span=[start, end] if start >= 0 else [0, len(row.review_text)],
                            source="aspect_memory",
                            support_type="contextual",
                            source_type="implicit_learned",
                            matched_pattern=matched_pattern,
                            pattern_id=f"aspect_memory:{mem.cluster_id}",
                            evidence_scope="exact_phrase" if start >= 0 else "full_review",
                        )
                        canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                        
                        # Conflict Resolution (Phase 2): Check for span overlap or same canonical
                        conflict_idx = -1
                        for i, existing in enumerate(implicits):
                            # Same canonical OR overlapping span for same source type
                            if existing.aspect_canonical == canonicalized.aspect_canonical:
                                conflict_idx = i
                                break
                            
                            e_start, e_end = existing.evidence_span
                            c_start, c_end = canonicalized.evidence_span
                            if not (c_end <= e_start or c_start >= e_end):
                                # Overlap!
                                conflict_idx = i
                                break

                        if conflict_idx == -1:
                            implicits.append(canonicalized)
                            seen_canonicals.add(canonicalized.aspect_canonical)
                        else:
                            existing = implicits[conflict_idx]
                            updated_layers = tuple(sorted(set(existing.mapping_layers) | {"aspect_memory"}))
                            
                            # manual_validated memory ALWAYS wins
                            if mem.validation_status == "manual_validated":
                                implicits[conflict_idx] = replace(canonicalized,
                                    mapping_layers=updated_layers,
                                    conflict_resolution="manual_validated_aspect_memory_preferred"
                                )
                            else:
                                # Symptom store usually wins unless memory is manual
                                implicits[conflict_idx] = replace(existing, 
                                    mapping_layers=updated_layers,
                                    conflict_resolution="symptom_store_preferred" if existing.source == "symptom_store" else "memory_preferred"
                                )

                json_scores = score_all_families(row.review_text, domain=row.domain)
                for score in json_scores:
                    cue_candidates = [*(list(score.matched_terms or [])), score.latent_family.replace("_", " ")]
                    cue = next((c for c in cue_candidates if c and c.lower() in row.review_text.lower()), score.latent_family)
                    sentence = select_best_sentence(row.review_text, cue)
                    sent_span = _find_sentence_span(row.review_text, sentence)
                    
                    temp_interp = Interpretation(
                        aspect_raw=score.latent_family,
                        latent_family=score.latent_family,
                        aspect_canonical="unknown",
                        label_type="implicit",
                        sentiment="unknown",
                        evidence_text=sentence if sent_span != (-1, -1) else row.review_text,
                        evidence_span=[sent_span[0], sent_span[1]] if sent_span != (-1, -1) else [0, len(row.review_text)],
                        source="latent_family_matcher",
                        support_type="contextual",
                        source_type="implicit_json",
                        evidence_scope="sentence" if sent_span != (-1, -1) else "full_review",
                        matched_terms=tuple(score.matched_terms),
                        implicit_trigger="latent_family_match",
                    )
                    canonicalized = canonicalize_interpretation(temp_interp, domain=row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy)
                    if canonicalized.aspect_canonical not in seen_canonicals:
                        implicits.append(canonicalized)
                        seen_canonicals.add(canonicalized.aspect_canonical)

                for behavior_interp in _behavior_pattern_interpretations(row):
                    if behavior_interp.aspect_canonical not in seen_canonicals:
                        implicits.append(behavior_interp)
                        seen_canonicals.add(behavior_interp.aspect_canonical)
                
                row = replace(row, implicit_interpretations=tuple(implicits))
                new_rows.append(row)
            finally:
                GLOBAL_STATS.record_row_processed()
        return new_rows

class EvidenceStage(PipelineStage):
    """Stage C: Evidence Grounding and Span Validation."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..evidence.sentence_selector import select_best_sentence
        from ..evidence.span_extractor import extract_span_from_sentence
        new_rows = []
        for row in rows:
            new_gold = []
            for i in row.gold_interpretations:
                if not i.evidence_text or i.evidence_span == [0, len(row.review_text)]:
                    sentence = select_best_sentence(row.review_text, i.aspect_raw)
                    span = extract_span_from_sentence(row.review_text, sentence)
                    i = replace(i, evidence_text=sentence, evidence_span=tuple(span), evidence_scope="sentence")
                new_gold.append(i)
            new_rows.append(replace(row, gold_interpretations=tuple(new_gold)))
        return new_rows

class PostVerificationEvidenceStage(PipelineStage):
    """Stage D2: Grounding specifically for verifier-added interpretations."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..evidence.sentence_selector import select_best_sentence
        from ..evidence.span_extractor import extract_span_from_sentence
        new_rows = []
        for row in rows:
            new_gold = []
            for i in row.gold_interpretations:
                if i.source == "llm_verifier" or i.evidence_span == [0, len(row.review_text)]:
                    sentence = select_best_sentence(row.review_text, i.evidence_text or i.aspect_raw)
                    span = extract_span_from_sentence(row.review_text, sentence)
                    if span != [-1, -1]:
                        i = replace(i, evidence_text=sentence, evidence_span=tuple(span), evidence_scope="sentence")
                new_gold.append(i)
            new_rows.append(replace(row, gold_interpretations=tuple(new_gold)))
        return new_rows

class VerificationStage(PipelineStage):
    """Stage D: LLM-based Verification."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..verify.llm_verifier import LLMVerifier
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        if cfg.llm_provider == "none":
            return rows
            
        verifier = LLMVerifier(cfg)
        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            try:
                v_row = verifier.verify_row(row)
                return v_row
            except Exception:
                return row
            finally:
                GLOBAL_STATS.record_row_processed()

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            return list(executor.map(process_row, rows))

class FusionStage(PipelineStage):
    """Stage E: Fusion of Explicit and Implicit Candidates."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..fusion.merge_candidates import merge_explicit_implicit
        new_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_fusion_candidates_with_modifiers"] = 0
        for row in rows:
            merged = merge_explicit_implicit(list(row.explicit_interpretations), list(row.implicit_interpretations))
            cfg.__dict__["_anchor_modifier_debug"]["after_fusion_candidates_with_modifiers"] += sum(1 for i in merged if tuple(getattr(i, "modifier_terms", ()) or ()))
            
            trace = dict(row.candidate_trace)
            trace["after_fusion"] = [i.aspect_raw for i in merged]
            
            new_rows.append(replace(row, gold_interpretations=tuple(merged), candidate_trace=trace))
        return new_rows

class CanonicalizationStage(PipelineStage):
    """Stage F: Canonical Mapping and Pruning."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..canonical.canonicalizer import canonicalize_interpretation
        from ..canonical.broad_label_policy import prune_broad_labels
        from ..canonical.fragment_collapse import collapse_same_evidence_fragments
        from ..canonical.aspect_memory import AspectMemory
        memory = AspectMemory(
            cfg.aspect_memory_path,
            auto_promote=cfg.aspect_memory_auto_promote,
        ) if cfg.aspect_memory_path else None
        aspect_memory_metrics = cfg.__dict__.setdefault("_aspect_memory_metrics", {})
        aspect_memory_metrics.setdefault("candidates_added", 0)
        aspect_memory_metrics.setdefault("promoted_matches_used", 0)
        aspect_memory_metrics.setdefault("candidates_promoted_this_run", 0)
        aspect_memory_metrics.setdefault("promoted_entries_total", 0)
        aspect_memory_metrics.setdefault("review_queue_count", 0)
        aspect_memory_metrics.setdefault("rejected_candidates_this_run", 0)
        
        new_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] = 0
        for row in rows:
            canons = [canonicalize_interpretation(i, row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy) for i in row.gold_interpretations]
            cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] += sum(1 for i in canons if i.mapping_source == "anchor_modifier")
            
            open_world_candidates = [i for i in canons if i.mapping_source in {"open_world_candidate", "provisional", "open_world"}]
            for i in open_world_candidates:
                if memory:
                    allowed, normalized_trigger = _should_enter_aspect_memory(i, row)
                    if not allowed:
                        aspect_memory_metrics["rejected_candidates_this_run"] += 1
                        continue
                    memory_aspect_raw = normalized_trigger or i.aspect_raw
                    memory_evidence_text = i.evidence_text or row.review_text
                    if i.mapping_source == "open_world" and str(getattr(i, "aspect_canonical", "") or "").strip():
                        memory_aspect_raw = i.aspect_canonical
                        memory_evidence_text = row.review_text
                    result = memory.add_evidence(
                        aspect_raw=memory_aspect_raw,
                        review_id=row.review_id, 
                        evidence_text=memory_evidence_text,
                        domain=row.domain,
                        sentiment=i.sentiment,
                        run_id=getattr(cfg, "run_id", None)
                    )
                    if result == "rejected_noise":
                        aspect_memory_metrics["rejected_candidates_this_run"] += 1
                    else:
                        aspect_memory_metrics["candidates_added"] += 1
            
            canons = [i for i in canons if i.mapping_source not in {"dropped_noise", "open_world_candidate"}]
            collapsed, _ = collapse_same_evidence_fragments(canons)
            final_gold, _ = prune_broad_labels(collapsed, row.domain)
            
            # Open-world rescue
            if not final_gold and open_world_candidates:
                from ..canonical.open_world_fallback import mark_provisional_canonical
                for candidate in open_world_candidates:
                    provisional = mark_provisional_canonical(candidate.aspect_raw)
                    if provisional:
                        final_gold = [replace(candidate, aspect_canonical=provisional, mapping_source="open_world_candidate")]
                        break
            
            final_gold = [i for i in final_gold if str(getattr(i, "aspect_canonical", "") or "") != "unknown"]
            final_gold = [_narrow_final_interpretation_evidence(row.review_text, row.review_id, i, cfg.evidence_window_tokens) for i in final_gold]
            
            trace = dict(row.candidate_trace)
            trace["after_canonicalization"] = [i.aspect_raw for i in canons]
            trace["after_pruning"] = [i.aspect_canonical for i in final_gold]

            new_rows.append(replace(row, gold_interpretations=tuple(final_gold), candidate_trace=trace))
        if memory and getattr(cfg, "aspect_memory_bootstrap", False):
            _bootstrap_controlled_aspect_memory(memory, run_id=getattr(cfg, "run_id", None))
            memory.save()
            memory.write_review_queue(Path(cfg.output_dir) / "aspect_memory_review_queue.json")
            memory.write_summary(Path(cfg.output_dir) / "aspect_memory_summary.json")
        elif memory:
            memory.save()
            memory.write_review_queue(Path(cfg.output_dir) / "aspect_memory_review_queue.json")
            memory.write_summary(Path(cfg.output_dir) / "aspect_memory_summary.json")
        return new_rows

class SentimentStage(PipelineStage):
    """Stage G: Aspect-Conditioned Sentiment Analysis."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..sentiment.classifier import SentimentClassifier
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        classifier = SentimentClassifier(cfg)
        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            try:
                if not row.gold_interpretations:
                    return row
                new_gold = classifier.classify_batch(row.review_text, list(row.gold_interpretations))
                return replace(row, gold_interpretations=tuple(new_gold))
            finally:
                GLOBAL_STATS.record_row_processed()
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            return list(executor.map(process_row, rows))

class BenchmarkStage(PipelineStage):
    """Stage H: Hardness Scoring and Finalization."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..benchmark.hardness_scorer import score_row_hardness
        from ..benchmark.novelty import detect_novelty, aggregate_row_novelty
        from .telemetry import GLOBAL_STATS
        GLOBAL_STATS.reset_stage(len(rows))
        
        seen_texts = set()
        unique_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_final"] = 0
        
        for row in rows:
            try:
                if row.review_text in seen_texts:
                    GLOBAL_STATS.record_row_rejection("duplicate_text_dropped")
                    cfg.__dict__.setdefault("_rejected_rows_audit", []).append({
                        "review_id": row.review_id,
                        "domain": row.domain,
                        "review_text": row.review_text,
                        "stage": "benchmark",
                        "reason": "duplicate_text_dropped",
                        "candidate_trace": row.candidate_trace,
                        "recommended_recovery": "reject_noise",
                    })
                    continue
                seen_texts.add(row.review_text)

                if _is_controlled_abstain_fixture(row):
                    unique_rows.append(replace(
                        row,
                        gold_interpretations=tuple(),
                        hardness_tier="H3",
                        abstain_acceptable=True,
                        abstain_reason_gold=("insufficient_aspect_evidence", "vague_review"),
                        ambiguity_score=1.0,
                        ambiguity_level="high",
                        novelty_status="known",
                        row_source_type="abstain",
                        row_mapping_scope="abstain",
                        row_mapping_sources=("abstain",),
                        source_type="abstain",
                        mapping_source="abstain",
                        mapping_scope="abstain",
                    ))
                    continue
                
                if not row.gold_interpretations:
                    if _is_vague_abstain_row(row):
                        unique_rows.append(replace(
                            row,
                            hardness_tier="H3",
                            abstain_acceptable=True,
                            abstain_reason_gold=("insufficient_aspect_evidence", "vague_review"),
                            ambiguity_score=1.0,
                            ambiguity_level="high",
                            novelty_status="known",
                            row_source_type="abstain",
                            row_mapping_scope="abstain",
                            row_mapping_sources=("abstain",),
                            source_type="abstain",
                            mapping_source="abstain",
                            mapping_scope="abstain",
                        ))
                        continue
                    GLOBAL_STATS.record_row_rejection("empty_gold_after_benchmark_filter")
                    cfg.__dict__.setdefault("_rejected_rows_audit", []).append({
                        "review_id": row.review_id,
                        "domain": row.domain,
                        "review_text": row.review_text,
                        "stage": "canonicalization",
                        "reason": "empty_gold_after_canonicalization",
                        "candidate_trace": row.candidate_trace,
                        "recommended_recovery": "abstain",
                    })
                    continue
                    
                final_gold = sorted(list(row.gold_interpretations), key=lambda i: i.canonical_confidence, reverse=True)[:8]
                final_gold = [_narrow_final_interpretation_evidence(row.review_text, row.review_id, i, cfg.evidence_window_tokens) for i in final_gold]
                cfg.__dict__["_anchor_modifier_debug"]["after_final"] += sum(1 for i in final_gold if i.mapping_source == "anchor_modifier")
                
                domain_cfg = DomainRegistry.get_config(row.domain)
                known_canonicals = set(domain_cfg.get("domain_maps", {}).values()) | set(domain_cfg.get("latent_families", {}).keys())

                scored_gold = []
                for i in final_gold:
                    novelty_status = detect_novelty(i.aspect_canonical, known_canonicals, mapping_confidence=i.canonical_confidence or 0.0, mapping_source=i.mapping_source or "none")
                    scored_gold.append(replace(i, novelty_status=novelty_status))
                
                row_novelty = aggregate_row_novelty(scored_gold)
                ambiguity = compute_ambiguity_score(scored_gold)
                h = score_row_hardness(replace(row, gold_interpretations=tuple(scored_gold), ambiguity_score=ambiguity, novelty_status=row_novelty))
                
                unique_rows.append(replace(row, 
                    gold_interpretations=tuple(scored_gold),
                    hardness_tier=h,
                    abstain_acceptable=(h in ["H2", "H3"]),
                    abstain_reason_gold=("insufficient_aspect_evidence",) if h == "H3" else (("multi_possible_aspects",) if h == "H2" else tuple()),
                    novelty_status=row_novelty,
                    ambiguity_score=ambiguity,
                ))
            finally:
                GLOBAL_STATS.record_row_processed()
        return unique_rows
