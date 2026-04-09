# Plan: Multi-Aspect Aspect Extraction (MAAE)

> Source PRD: [prd_dataset_quality.md](file:///c:/Users/MONISH/Desktop/GitHub%20Repo/agent_docs/ReviewOp/prd_dataset_quality.md) & Grill-me results

## Architectural decisions

- **Grounding Layer**: Vector-Lexical Hybrid (Noun chunks + Embedding distance).
- **Segmentation Layer**: Discourse-Aware (Splitting on coordinating conjunctions).
- **Service Dependency**: ProtoNet service must be active during dataset building.
- **Model**: `HybridTextEncoder` (Vectorization) + `ProtoNet` (Classification).

---

## Phase 1: Pipeline Restoration (Tracer Bullet)

**User stories**: *"Generate accurate multi-label outputs for research-grade benchmarks"*.

### What to build
Restore the `dataset_builder` end-to-end path by relaxing the strict research profile constraints. This allows for rapid iteration and validation of the logic changes in later phases.

### Acceptance criteria
- [ ] `train_target_min` lowered to 500 in `cfg`.
- [ ] `sample_rate` increased to 0.4.
- [ ] `python build_dataset.py` completes successfully without `train_target_size_within_range` blocking.

---

## Phase 2: Hybrid Vector Grounding

**User stories**: *"Prioritize label quality over coverage"*, *"Move beyond static keyword defaults"*.

### What to build
Integrate SpaCy-based candidate extraction and Vector Similarity mapping into `implicit_pipeline.py`. This removes the reliance on the hardcoded `LATENT_ASPECT_RULES`.

### Acceptance criteria
- [ ] `extract_open_aspects` integrated into `build_implicit_row`.
- [ ] Grounding matches validated via Vector Distance to centroids.
- [ ] Specialized terms (e.g., "Moules") correctly mapped to `sensory quality`.

---

## Phase 3: Discourse-Aware Splitting & Multi-Label

**User stories**: *"Multi-aspect extraction for complex, multi-sentence reviews"*.

### What to build
Enhance the segmenter to isolate concurrent aspects and update the inference service to return multiple labels per segment if they are within the confidence margin.

### Acceptance criteria
- [ ] `split_clauses` handles 'and', 'but', 'however'.
- [ ] `inference_service` returns a list of labels per segment.
- [ ] Row 10 ("Moules... lobster ravioli...") produces two distinct `sensory quality` spans.

---

## Phase 4: Specificity & Conflict Resolution

**User stories**: *"Exclude borderline rows from training"*, *"Resolve inconsistencies"*.

### What to build
Implement the quality-first filters that suppress generic labels when specific ones exist and flag domain mismatches (e.g., Food word + Service sentiment).

### Acceptance criteria
- [ ] `general` label suppressed when specific aspects are found.
- [ ] Conflict resolver flags "Domain Clashes".
- [ ] Verification on Row 2 (Waitstaff + Food keywords) passes.
