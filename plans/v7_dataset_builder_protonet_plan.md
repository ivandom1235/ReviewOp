# Plan: V7 Dataset Builder and ProtoNet Stabilization

> Source PRD: [Project_II_V7_DatasetBuilder_ProtoNet_Plan.md](/C:/Users/MONISH/Desktop/GitHub%20Repo/ReviewOp/Project_II_V7_DatasetBuilder_ProtoNet_Plan.md)

## Architectural decisions

These are the durable decisions that should remain stable across the implementation:

- **Builder orchestration stays centralized**: `dataset_builder` should keep one top-level pipeline entrypoint, but policy decisions must move into smaller focused modules.
- **Benchmark remains strict**: the gold benchmark should preserve the strongest grounding and leakage rules.
- **Train export is usefulness-aware**: train selection may accept weaker grounded rows if they are useful for coverage, balance, or hard-case learning.
- **Selective states are first-class**: abstain, novelty, silver, train-keep, and hard-reject are separate outcomes and should be reported separately.
- **Deduplication is two-stage**: cheap exact/fuzzy dedup first, semantic dedup second, with split-aware limits.
- **ProtoNet remains a component, not the paper center**: it should support selective inference, abstain, and novelty routing, but the benchmark contract drives the research claim.
- **TDD is mandatory for each slice**: every phase begins with a failing test that exercises public behavior before code changes are accepted.
- **No silent policy fallback**: any recovery, promotion, or rejection path must record why it happened in the exported report or sidecar.

## Current code realities to preserve

The plan should fit the current codebase rather than replacing it wholesale:

- `dataset_builder/code/build_dataset.py` is already the orchestration-heavy entrypoint and should be reduced further, not duplicated.
- `dataset_builder/code/contracts.py` already defines the central builder dataclasses and should be extended rather than replaced.
- `dataset_builder/code/splitter.py` already handles grouped and leakage-aware splitting, so v7 should extend its policy inputs instead of inventing a new split engine.
- `protonet/code/evaluator.py` and `protonet/code/runtime_infer.py` already compute abstain and novelty signals, so v7 should tighten calibration, not rewrite the whole inference path.
- Existing tests are sparse but focused on behavior; new tests should stay in that style and target public outcomes, not internals.

---

## Phase 1: Make the builder state and reports explicit

**User stories**: builder observability, maintainable orchestration, explainable failures, stable output contracts.

### What to build

Create the minimum state model needed to make v7 debuggable and auditable:

- formal row lifecycle states for prepared/scored/bucketed/promoted/rejected outcomes,
- explicit decision objects for quality, recovery, and export selection,
- report payloads that record why a row was accepted, promoted, deferred, or rejected,
- a compact profiling sidecar for stage timing and cache usage.

The goal of this phase is not new selection behavior. The goal is to make current behavior legible and machine-checkable before any policy changes land.

### TDD checkpoints

1. Add tests for row contract serialization and lifecycle state round-tripping.
2. Add tests that a report payload includes decision reasons for accepted and rejected rows.
3. Add tests for a profiling sidecar shape that reports stage timing and counts.

### Acceptance criteria

- [ ] A row can be represented through the full lifecycle without losing its state.
- [ ] Decision objects expose a stable, serializable shape.
- [ ] Build reports include rejection and promotion reasons instead of only aggregate counts.
- [ ] Profiling output can be generated without depending on model internals.

---

## Phase 2: Separate benchmark policy from train policy

**User stories**: preserve a strict benchmark, avoid train underfill, keep rare domains and sentiments alive, make silver rows reusable.

### What to build

Move target-specific selection rules into distinct policy paths:

- a strict benchmark-gold path,
- a moderate benchmark-silver path,
- a permissive train-keep path,
- a hard-reject path for malformed or unsupported rows.

This phase should also make usefulness part of selection so the train path can keep grounded rows that are not gold-worthy but still improve coverage. The selection logic should still honor domain compatibility, evidence quality, and leakage safety.

### TDD checkpoints

1. Add tests proving the same candidate can be routed differently for gold benchmark, silver benchmark, and train export.
2. Add tests showing borderline grounded rows are kept for train when they help coverage, but rejected from gold.
3. Add tests that hard reject remains reserved for malformed or unrecoverable rows.

### Acceptance criteria

- [ ] Gold benchmark selection remains stricter than silver.
- [ ] Silver selection produces rows that are still useful downstream.
- [ ] Train export no longer depends on the exact same threshold set as gold selection.
- [ ] Rejection reasons distinguish malformed rows from merely borderline rows.

---

## Phase 3: Add multi-objective scoring and shortage-aware promotion

**User stories**: reduce over-rejection, improve sentiment balance, preserve telecom and minority families, make recovery usefulness-aware.

### What to build

Introduce a row scoring layer with three stable outputs:

- **quality**: grounding strength, support reliability, ontology compatibility, domain compatibility,
- **usefulness**: rare domain contribution, rare sentiment contribution, abstain value, hard-case value, diversity value,
- **redundancy**: exact duplication risk, semantic duplication risk, cluster saturation risk.

Then use those scores to drive:

- recovery ranking,
- train top-up selection,
- silver promotion,
- family floor enforcement late in the pipeline,
- shortage-aware boosts for underrepresented domains and sentiment classes.

The important constraint is that scoring should rank candidates for different outputs differently. A high-quality row is not always the most useful row, and the plan should reflect that explicitly.

### TDD checkpoints

1. Add tests for the three score types on a known candidate set.
2. Add tests showing that top-up prefers a useful borderline row over a redundant but slightly cleaner row when the target is short on that category.
3. Add tests that family floor checks still protect underrepresented groups after promotion and top-up.

### Acceptance criteria

- [ ] Quality, usefulness, and redundancy are separate concepts in code and reports.
- [ ] Train top-up behavior changes when the run is short on a target class or domain.
- [ ] Telecom and minority-family rows are not lost late in the pipeline.
- [ ] Sentiment balance checks run after promotion, not only before.

---

## Phase 4: Make deduplication semantic and split-aware

**User stories**: lower duplicate logical row rate, prevent cluster flooding, improve benchmark diversity, keep novelty leakage down.

### What to build

Replace one-dimensional duplicate handling with a two-stage dedup pipeline:

- stage A: exact and fuzzy matching over normalized review/evidence signatures,
- stage B: semantic clustering over review text, evidence text, and aspect-evidence pairs.

Then apply split-aware caps:

- per semantic cluster,
- per split,
- per domain family.

Selection from duplicate clusters should prefer rows that add diversity in domain, sentiment, ambiguity type, and support type.

### TDD checkpoints

1. Add tests for exact duplicate collapse using normalized review signatures.
2. Add tests for semantic near-duplicate grouping using the dedup abstraction.
3. Add tests that split-aware limits prevent one cluster from dominating multiple splits.

### Acceptance criteria

- [ ] Duplicate handling is no longer only an after-the-fact filter.
- [ ] Dedup decisions are explainable at the cluster level.
- [ ] Selection from a cluster can prefer the row that improves dataset diversity.
- [ ] Split leakage checks incorporate dedup cluster identity.

---

## Phase 5: Stabilize abstain and novelty as calibrated selective decisions

**User stories**: make abstain explicit, make novelty auditable, support selective inference in ProtoNet, improve boundary metrics.

### What to build

Formalize abstain and novelty into explicit decision paths:

- abstain reason taxonomy,
- calibrated thresholding for known/boundary/novel routing,
- validation-gated novelty calibration,
- leakage audits for novelty cluster IDs, aliases, and support text.

This phase should align builder outputs and ProtoNet runtime behavior so the same decision vocabulary appears in both systems. The model can still be simple, but the calibration contract must be consistent.

### TDD checkpoints

1. Add tests that abstain reasons are preserved in exported benchmark rows and evaluation rows.
2. Add tests that novelty calibration is skipped when validation data is insufficient.
3. Add tests that runtime and evaluator novelty scoring use the same threshold interpretation.

### Acceptance criteria

- [ ] Abstain decisions include a stable reason label.
- [ ] Novelty routing only calibrates when validation support is sufficient.
- [ ] Runtime and evaluation use the same selective decision contract.
- [ ] Boundary cases are reported separately from obvious known or obvious novel cases.

---

## Phase 6: Improve synthetic and weak-supervision paths

**User stories**: fill coverage gaps, preserve rare domains, reduce train underfill without polluting benchmark, support challenge-aware augmentation.

### What to build

Treat synthetic and weak-supervision data as a controlled coverage tool, not a generic augmenter:

- explicit-to-implicit paraphrase bank,
- challenge-aware synthetic generation for shortage categories,
- dedup-aware acceptance for synthetic rows,
- provenance fields that identify generator source and intended role,
- optional weak-supervision acceptance when grounding is adequate.

This phase should make the synthetic path useful for train and silver recovery while keeping the benchmark strict.

### TDD checkpoints

1. Add tests that synthetic rows record provenance and intended use.
2. Add tests that duplicated synthetic outputs are rejected or downgraded.
3. Add tests that challenge-aware synthesis targets documented shortages.

### Acceptance criteria

- [ ] Synthetic rows cannot silently enter the benchmark gold path.
- [ ] Generated rows preserve provenance and selection intent.
- [ ] Coverage gaps can be filled without inflating duplicate clusters.
- [ ] Weak supervision is distinguishable from gold evidence in reports.

---

## Phase 7: Tighten ProtoNet into a selective-inference component

**User stories**: keep ProtoNet as a baseline/component, improve evidence-conditioned scoring, support novelty and abstain, avoid closed-set-only evaluation.

### What to build

Refine ProtoNet around the selective-inference contract already emerging in the codebase:

- evidence-conditioned support/query construction,
- aspect-only and joint-label modes where appropriate,
- better calibration for abstain and novelty thresholds,
- benchmark-aware evaluation that reports coverage-risk and grouped/domain holdout results,
- runtime export that preserves decision bands and candidate metadata.

The focus is not to create a new classifier family. The focus is to make the current ProtoNet path consistent with the benchmark design and selective decisioning.

### TDD checkpoints

1. Add tests for evidence-conditioned query construction from benchmark instances.
2. Add tests that grouped/domain-holdout metrics are emitted alongside standard metrics.
3. Add tests that runtime exports preserve abstain and novel candidate metadata.

### Acceptance criteria

- [ ] ProtoNet remains usable as a few-shot baseline.
- [ ] ProtoNet reports selective metrics, not only exact accuracy.
- [ ] Runtime and evaluation disagree less on novelty and abstain behavior.
- [ ] The exported bundle remains compatible with backend loading.

---

## Phase 8: Shrink orchestration and stabilize release reporting

**User stories**: keep the pipeline maintainable, make small runs representative, make release outputs interpretable, support repeated runs.

### What to build

Finish the refactor by pushing orchestration back to the top level and leaving policy in focused modules:

- keep the top-level dataset build path orchestration-only,
- preserve the current benchmark artifact contract,
- emit clear run summaries with gold/silver/train/reject/recovery counts,
- add a compact “why not promoted?” sidecar,
- make profiling and cache reuse visible in the final report.

This phase is the integration pass: it should not introduce new selection ideas. It should make the new policy layers visible, stable, and easy to operate.

### TDD checkpoints

1. Add a high-level pipeline test that the run emits all required artifacts.
2. Add a report test that the summary distinguishes gold, silver, train, reject, and recovery outcomes.
3. Add a regression test that a small deterministic run still produces a representative output set.

### Acceptance criteria

- [ ] The orchestration layer is thin and readable.
- [ ] Final reports explain promotion failures and shortlist reasons.
- [ ] Existing benchmark artifact paths remain valid.
- [ ] A deterministic small run can still complete without special-case manual intervention.

---

## Testing strategy

Use behavior-focused tests only:

- test public contracts and exported artifacts,
- test selection and promotion through stable inputs and outputs,
- avoid mocking internal helpers unless there is no public seam,
- prefer small deterministic fixtures over broad synthetic fixtures,
- add regression tests for the exact failure mode each phase is meant to remove.

Coverage priorities:

1. row contracts and report shapes,
2. policy routing and top-up behavior,
3. dedup and split safety,
4. selective inference and calibration,
5. orchestration/report integration.

## Out of scope

- Rewriting the entire dataset builder into a new package tree in one pass.
- Replacing ProtoNet with a different model family.
- Changing backend or frontend behavior beyond what is needed to preserve bundle compatibility.
- Introducing new external dependencies unless a phase proves they are necessary.
- Making the benchmark less strict in the name of coverage.

## Implementation order

The recommended order is:

1. state and reporting,
2. policy separation,
3. scoring and promotion,
4. deduplication,
5. abstain and novelty calibration,
6. synthetic/weak supervision,
7. ProtoNet selective-inference alignment,
8. orchestration cleanup and release polish.

