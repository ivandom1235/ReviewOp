from __future__ import annotations
import abc
from abc import ABC, abstractmethod
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from ..schemas.benchmark_row import BenchmarkRow
from ..schemas.interpretation import Interpretation
from ..config import BuilderConfig
from ..explicit.phrase_rules import extract_noun_chunks, extract_dependency_phrases
from ..explicit.phrase_cleaning import is_noisy_label
from ..implicit.symptom_store import SymptomPatternStore
from ..implicit.latent_families import score_family_match
from ..canonical.domain_registry import DomainRegistry
from ..canonical.domain_maps import lookup_domain_map
from ..benchmark.novelty import detect_novelty, aggregate_row_novelty

class PipelineStage(ABC):
    @abstractmethod
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        """Process a list of rows and return the modified list."""
        pass

def _extract_for_row(row: BenchmarkRow) -> BenchmarkRow:
    """Helper function for ProcessPoolExecutor."""
    chunks = extract_noun_chunks(row.review_text)
    phrases = extract_dependency_phrases(row.review_text)
    
    new_interps = []
    for c in chunks:
        new_interps.append(Interpretation(
            aspect_raw=c["text"],
            aspect_canonical=c["aspect_anchor"], # Fallback for legacy
            latent_family="unknown",
            label_type="explicit",
            sentiment="unknown",
            evidence_text=c["text"],
            evidence_span=c["span"],
            source="spacy_noun_chunk",
            support_type="exact",
            source_type="explicit",
            aspect_anchor=c["aspect_anchor"],
            modifier_terms=c["modifier_terms"],
            anchor_source=c["anchor_source"],
            evidence_scope="exact_phrase"
        ))
        
    for p in phrases:
        new_interps.append(Interpretation(
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
            modifier_terms=p["modifier_terms"],
            anchor_source=p["anchor_source"],
            evidence_scope="exact_phrase"
        ))
    
    return replace(
        row,
        explicit_interpretations=tuple(new_interps)
    )

class ExtractionStage(PipelineStage):
    """Stage A: Explicit Extraction using spaCy and Multiprocessing."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        if not rows:
            return rows
        max_workers = getattr(cfg, "max_workers", 4)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            processed = list(executor.map(_extract_for_row, rows))
        return processed

class InferenceStage(PipelineStage):
    """Stage B: Implicit Inference (Learned Patterns + JSON Fallback)."""
    _store_cache: dict[str, SymptomPatternStore] = {}

    def _get_store(self, path: str | None) -> SymptomPatternStore | None:
        # Default fallback if no path provided
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
        except Exception:
            return None

    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        store = self._get_store(cfg.symptom_store_path)
        new_rows = []
        
        for row in rows:
            implicits = []
            matched_any_learned = False
            seen_canonicals = set()
            
            # 1. Learned Detection
            if store:
                matches = store.match(row.review_text, domain=row.domain)
                for match in matches:
                    matched_any_learned = True
                    family_score = score_family_match(match.matched_pattern, domain=row.domain)
                    latent_family = match.latent_family or family_score.latent_family
                    
                    # PRECEDENCE FLIP: Store wins over domain map
                    res = lookup_domain_map(row.domain, match.matched_pattern)
                    canonical = match.aspect_canonical or res.aspect_canonical or "unknown"
                    mapping_source = "symptom_store"
                    if res.mapping_confidence > 0:
                        mapping_source = "symptom_store+domain_json"

                    implicits.append(Interpretation(
                        aspect_raw=match.matched_pattern,
                        aspect_canonical=canonical,
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
                        mapping_source=mapping_source,
                        source_type="implicit_learned"
                    ))
                    seen_canonicals.add(canonical)
            
            # 3. Fallback JSON Inference
            if not matched_any_learned:
                score = score_family_match(row.review_text, domain=row.domain)
                if score.latent_family != "unknown":
                    implicits.append(Interpretation(
                        aspect_raw=score.latent_family,
                        aspect_canonical=score.latent_family,
                        latent_family=score.latent_family,
                        label_type="implicit",
                        sentiment="unknown",
                        evidence_text=row.review_text,
                        evidence_span=[0, len(row.review_text)],
                        source="latent_family_matcher",
                        support_type="contextual",
                        source_type="implicit_json",
                        evidence_scope="full_review"
                    ))
            
            row = replace(row, implicit_interpretations=tuple(implicits))
            new_rows.append(row)
            
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
                if not i.evidence_text or i.evidence_span == [-1, -1] or i.evidence_span == (0, 0) or i.evidence_span == [0, 0]:
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
                # Only ground if it's from the verifier or has missing span
                if i.source == "llm_verifier" or i.evidence_span == [0, len(row.review_text)]:
                    sentence = select_best_sentence(row.review_text, i.evidence_text or i.aspect_raw)
                    span = extract_span_from_sentence(row.review_text, sentence)
                    if span == [-1, -1]:
                        # If we can't ground it, drop it (as per design)
                        continue
                    i = replace(i, evidence_text=sentence, evidence_span=tuple(span), evidence_scope="sentence")
                new_gold.append(i)
            new_rows.append(replace(row, gold_interpretations=tuple(new_gold)))
        return new_rows

class VerificationStage(PipelineStage):
    """Stage D: LLM-based Verification (Keep/Drop/Merge)."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..verify.openai_verifier import OpenAIVerifier
        
        if cfg.llm_provider == "none":
            new_rows = []
            for row in rows:
                filtered_gold = [i for i in row.gold_interpretations if not is_noisy_label(i.aspect_raw)]
                new_rows.append(replace(row, gold_interpretations=tuple(filtered_gold)))
            return new_rows

        verifier = OpenAIVerifier(cfg)

        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            # First pass: deterministic noisy filter
            filtered_gold = [i for i in row.gold_interpretations if not is_noisy_label(i.aspect_raw)]
            if not filtered_gold:
                return replace(row, gold_interpretations=tuple())
            
            try:
                v_row = verifier.verify_row(replace(row, gold_interpretations=tuple(filtered_gold)))
                # Second pass: ensure NO noisy labels survive
                final_gold = [i for i in v_row.gold_interpretations if not is_noisy_label(i.aspect_raw)]
                return replace(v_row, gold_interpretations=tuple(final_gold))
            except Exception:
                return replace(row, gold_interpretations=tuple(filtered_gold))

        with ThreadPoolExecutor(max_workers=20) as executor:
            return list(executor.map(process_row, rows))

class FusionStage(PipelineStage):
    """Stage E: Fusion of Explicit and Implicit Candidates."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..fusion.merge_candidates import merge_explicit_implicit
        return [
            replace(row, gold_interpretations=tuple(merge_explicit_implicit(list(row.explicit_interpretations), list(row.implicit_interpretations))))
            for row in rows
        ]

class CanonicalizationStage(PipelineStage):
    """Stage F: Canonical Mapping and Pruning."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..canonical.canonicalizer import canonicalize_interpretation
        from ..canonical.broad_label_policy import prune_broad_labels
        from ..canonical.fragment_collapse import collapse_same_evidence_fragments
        
        new_rows = []
        for row in rows:
            # 1. Canonicalize
            canons = [canonicalize_interpretation(i, row.domain) for i in row.gold_interpretations]
            # 2. Collapse fragments
            collapsed, _ = collapse_same_evidence_fragments(canons)
            # 3. Prune broad labels
            final_gold, _ = prune_broad_labels(collapsed, row.domain)
            new_rows.append(replace(row, gold_interpretations=tuple(final_gold)))
        return new_rows

class SentimentStage(PipelineStage):
    """Stage G: Aspect-Conditioned Sentiment Analysis."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..sentiment.classifier import SentimentClassifier
        classifier = SentimentClassifier(cfg)
        
        def process_row(row: BenchmarkRow) -> BenchmarkRow:
            if not row.gold_interpretations:
                return row
            new_gold = classifier.classify_batch(row.review_text, list(row.gold_interpretations))
            return replace(row, gold_interpretations=tuple(new_gold))

        with ThreadPoolExecutor(max_workers=20) as executor:
            return list(executor.map(process_row, rows))

class BenchmarkStage(PipelineStage):
    """Stage H: Hardness Scoring and Finalization."""
    def process(self, rows: list[BenchmarkRow], cfg: BuilderConfig) -> list[BenchmarkRow]:
        from ..benchmark.hardness_scorer import score_row_hardness
        from ..benchmark.novelty import detect_novelty
        seen_texts = set()
        unique_rows = []
        
        for row in rows:
            # 1. Dedupe by text
            if row.review_text in seen_texts:
                continue
            seen_texts.add(row.review_text)
            
            # 2. Filter empty gold
            if not row.gold_interpretations:
                continue
                
            # 3. Cap interpretations
            final_gold = sorted(list(row.gold_interpretations), key=lambda i: i.canonical_confidence, reverse=True)[:8]
            
            # 4. Calculate Ambiguity Score
            sentiments = {i.sentiment for i in final_gold if i.sentiment != "unknown"}
            ambiguity = min(1.0, (len(final_gold) / 10.0) + (0.3 if len(sentiments) > 1 else 0.0))
            
            # 5. Determine Novelty Status
            domain_cfg = DomainRegistry.get_config(row.domain)
            # known_canonicals are the unique values (labels) in the domain maps and latent families
            known_from_map = set(domain_cfg.get("domain_maps", {}).values())
            known_from_families = set(domain_cfg.get("latent_families", {}).keys())
            known_canonicals = known_from_map | known_from_families
            
            scored_gold = []
            for i in final_gold:
                novelty_status = detect_novelty(
                    i.aspect_canonical, 
                    known_canonicals,
                    mapping_confidence=i.canonical_confidence or 0.0,
                    mapping_source=i.mapping_source or "none"
                )
                # Store novelty status in interpretation for aggregation
                i = replace(i, novelty_status=novelty_status)
                scored_gold.append(i)
            
            row_novelty = aggregate_row_novelty(scored_gold)
            
            unique_rows.append(replace(row, 
                gold_interpretations=tuple(scored_gold),
                hardness_tier=(h := score_row_hardness(replace(row, gold_interpretations=tuple(scored_gold)))),
                abstain_acceptable=(h in ["H2", "H3"]),
                novelty_status=row_novelty,
                ambiguity_score=ambiguity
            ))
            
        return unique_rows
