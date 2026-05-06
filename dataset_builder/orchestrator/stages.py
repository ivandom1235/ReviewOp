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
        
        new_rows = []
        cfg.__dict__.setdefault("_anchor_modifier_debug", {})
        cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] = 0
        for row in rows:
            canons = [canonicalize_interpretation(i, row.domain, domain_mode=cfg.domain_mode, provisional_policy=cfg.provisional_policy) for i in row.gold_interpretations]
            cfg.__dict__["_anchor_modifier_debug"]["after_canonicalization"] += sum(1 for i in canons if i.mapping_source == "anchor_modifier")
            
            open_world_candidates = [i for i in canons if i.mapping_source in {"open_world", "open_world_candidate", "provisional"}]
            for i in open_world_candidates:
                if memory:
                    memory.add_evidence(
                        aspect_raw=i.aspect_raw, 
                        review_id=row.review_id, 
                        evidence_text=i.evidence_text or row.review_text, 
                        domain=row.domain,
                        sentiment=i.sentiment,
                        run_id=getattr(cfg, "run_id", None)
                    )
            
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
        if memory:
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
                    continue
                seen_texts.add(row.review_text)
                
                if not row.gold_interpretations:
                    GLOBAL_STATS.record_row_rejection("empty_gold_after_benchmark_filter")
                    cfg.__dict__.setdefault("_rejected_rows_audit", []).append({
                        "review_id": row.review_id,
                        "domain": row.domain,
                        "review_text": row.review_text,
                        "stage": "canonicalization",
                        "reason": "empty_gold_after_canonicalization",
                        "candidate_trace": row.candidate_trace,
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
