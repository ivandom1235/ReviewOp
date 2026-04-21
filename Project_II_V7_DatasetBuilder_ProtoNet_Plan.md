# Project II — V7 Plan for Dataset Builder and ProtoNet

## 1. Purpose

This document combines the V7 direction for the **dataset_builder** and **ProtoNet** into one implementation and research plan.

The goal of V7 is to move from a pipeline that is already capable of producing clean, ambiguity-aware benchmark slices into a system that is:
- easier to maintain,
- less aggressive about discarding useful rows,
- more reliable on grouped and domain-aware evaluation,
- faster in repeated runs,
- more modular,
- and better aligned with the actual paper contribution.

The core V7 idea is:

> **Keep the gold benchmark strict, make the train path usefulness-aware, calibrate abstain and novelty, and split the builder into maintainable policy modules.**

---

## 2. What V6 achieved

V6 already established several strong foundations:
- benchmark-first dataset construction,
- grouped split hygiene and leakage auditing,
- evidence-grounded implicit extraction,
- ambiguity-aware rows with multi-gold interpretations,
- abstain-capable slices,
- novelty-capable slices,
- silver-tier recovery,
- strong reporting, manifests, and governance,
- a usable ProtoNet baseline and runtime path.

These are real strengths and should be preserved.

---

## 3. Main V6 problems that V7 must fix

The recurring issues across iterations were:

### Dataset / benchmark problems
- over-rejection of borderline but useful rows,
- too many selected rows deferred before final export,
- train export underfilling,
- train sentiment collapsing toward neutral,
- telecom and minority families being present upstream but disappearing downstream,
- duplicate logical row rate staying too high,
- implicit purity staying too low,
- top-up recovery being too weak or unstable,
- small research-profile runs becoming unrepresentative.

### Code / architecture problems
- `build_dataset.py` is too large and too policy-dense,
- benchmark filtering, train filtering, recovery, novelty, balancing, and export logic are too entangled,
- policy coupling makes threshold tuning fragile,
- promotion/debug labeling is sometimes hard to interpret,
- recovery logic exists but is still not sufficiently usefulness-aware.

### ProtoNet problems
- ProtoNet is useful, but should not define the paper,
- novelty routing support exists but calibration is fragile,
- evaluation often undersells the benchmark by treating the task too much like closed-set classification,
- ProtoNet is still too tightly thought of as a classifier rather than a scorer used inside selective inference.

---

## 4. V7 research and engineering position

V7 should remain centered on the actual Project II contribution:

### Main paper identity
- ambiguity-aware benchmark,
- evidence-grounded implicit interpretation,
- selective inference with abstention,
- grouped and domain-aware evaluation,
- novelty-aware extension as a later layer.

### Not the main paper identity
- ProtoNet alone,
- graph/dashboard/system layer alone,
- silver pool alone,
- LLM-only reasoning.

ProtoNet stays important, but as a **baseline and component**, not the headline novelty.

---

## 5. V7 design principles

### P1. Benchmark-first, but not benchmark-only
Gold benchmark rows must remain strict.
Train rows may be weaker than gold benchmark rows if they are still useful.

### P2. Quality and usefulness must both matter
Rows should be scored by:
- quality,
- usefulness,
- redundancy.

### P3. Hard rejection should be rare
Hard reject only truly bad rows.
Borderline rows should usually become silver, reviewable, or train-usable.

### P4. Abstain and novelty should be first-class states
Abstain-worthy rows are not low-quality accidents.
Novelty candidates are not automatically bad rows.

### P5. Separate policies by output target
A row should not be judged by the exact same logic for:
- benchmark gold,
- benchmark silver,
- train export,
- review queue,
- hard rejection.

### P6. Make small runs representative
Short laptop runs should still preserve the real behavior of the research profile.

---

## 6. V7 target architecture

V7 should split the builder into explicit subsystems.

```text
v7/
  dataset_builder/
    app/
      build_cli.py
      pipeline_runner.py
      run_experiment.py

    ingestion/
      io_utils.py
      schema_detect.py
      language_utils.py
      source_registry.py

    extraction/
      implicit_pipeline.py
      explicit_features.py
      coref.py
      llm_utils.py
      aspect_registry.py

    curation/
      row_contracts.py
      row_scoring.py
      row_bucketing.py
      row_recovery.py
      row_promotion.py
      deduplication.py
      benchmark_balance.py
      family_floor.py

    splitting/
      splitter.py
      leakage_audit.py
      novelty_split_guard.py

    benchmark/
      benchmark_instances.py
      benchmark_export.py
      benchmark_protocols.py
      abstain_logic.py
      novelty_sidecar.py

    trainset/
      train_export.py
      sentiment_balance.py
      train_topup.py
      train_floor.py

    evaluation/
      evaluation.py
      robustness_eval.py
      quality_metrics.py
      selective_metrics.py
      novelty_metrics.py

    governance/
      experiment_policy.py
      governance.py
      report_blockers.py
      report_payload.py
      report_context.py

    reporting/
      exporters.py
      sidecars.py
      markdown_summary.py
      analyze_reports.py

    utils/
      config.py
      research_stack.py
      pipeline_state.py
      pipeline_helpers.py
```

---

## 7. Core code-architecture refactor plan

## 7.1 Shrink `build_dataset.py`

### V6 problem
`build_dataset.py` currently mixes:
- orchestration,
- scoring,
- review filtering,
- salvage,
- top-up,
- balancing,
- novelty routing,
- reporting hooks,
- export decisions.

### V7 target
Make it orchestration-only:
1. load config,
2. call pipeline stages,
3. gather outputs,
4. call reporting/export,
5. exit.

### Target
- orchestration file <= 900 lines,
- no benchmark policy logic in orchestration,
- no train top-up logic in orchestration,
- no novelty routing heuristics in orchestration.

---

## 7.2 Add a row lifecycle state machine

Each row should move through explicit states:
- `raw_loaded`
- `prepared`
- `implicit_scored`
- `grounded`
- `quality_scored`
- `bucketed`
- `dedup_checked`
- `split_assigned`
- `benchmark_gold`
- `benchmark_silver`
- `train_keep`
- `review_queue`
- `hard_reject`
- `promoted_to_train`
- `promoted_to_benchmark`

This makes debugging, reporting, and threshold tuning much easier.

---

## 7.3 Formalize row contracts

Introduce explicit types for:
- `CandidateRow`
- `GroundedRow`
- `QualityDecision`
- `RecoveryDecision`
- `BenchmarkInstance`
- `TrainExample`
- `NoveltyRecord`

These should be dataclasses or Pydantic models.

---

## 8. V7 dataset pipeline redesign

## 8.1 Use multi-objective row scoring

Every row should receive at least three scores.

### A. Quality score
Measures:
- grounding strength,
- evidence precision,
- support reliability,
- ontology compatibility,
- domain compatibility,
- implicit purity.

### B. Usefulness score
Measures:
- rare domain contribution,
- rare sentiment contribution,
- abstain contribution,
- novelty contribution,
- hard-case contribution,
- benchmark diversity contribution,
- family floor contribution.

### C. Redundancy score
Measures:
- semantic duplication,
- cluster saturation,
- cross-split similarity risk,
- template overuse.

Rows should then be ranked differently depending on target output.

---

## 8.2 Separate output policies by target

### Gold benchmark policy
Very strict.
- best support types,
- no explicit contamination,
- stronger domain compatibility,
- stronger dedup,
- best grounding only.

### Silver benchmark policy
Moderately strict.
- allows mild domain soft mismatch,
- allows weaker but grounded support,
- preserves rare domains and rare sentiments,
- preserves abstain-worthy rows.

### Train export policy
More permissive.
- allows symptom-based and paraphrastic support,
- allows mild soft mismatch,
- allows moderate confidence if usefulness is high,
- prioritizes coverage and training value over benchmark purity.

### Hard reject policy
Reserved for:
- malformed rows,
- nonsense,
- unsupported after recovery,
- severe mismatch with weak evidence,
- unrecoverable duplicates.

---

## 8.3 Make silver rows genuinely useful

V6 introduced silver, which was the right move. V7 should complete the design by supporting:
- `silver -> train_keep`
- `silver -> benchmark_silver`
- `silver -> abstain_bucket`
- `silver -> novelty_bucket`

Promotion should be shortage-aware. If the run is short on:
- telecom,
- negative sentiment,
- abstain rows,
- novelty rows,
- H2/H3 hard cases,
then relevant silver rows should receive a boost.

---

## 8.4 Preserve family floors late in the pipeline

Family floors should not only be checked early.
They should be re-applied:
1. after candidate selection,
2. after promotion,
3. after final export.

This is especially important for:
- telecom,
- electronics,
- minority service domains,
- underrepresented sentiment classes.

---

## 9. Deduplication redesign

This is one of the most important V7 changes.

## 9.1 Two-stage deduplication

### Stage A — cheap exact/fuzzy dedup
Use:
- normalized review hash,
- evidence hash,
- n-gram overlap,
- template or source family IDs.

### Stage B — semantic deduplication
Use embeddings on:
- review text,
- evidence text,
- aspect+evidence pairs,
- novelty alias text when relevant.

Cluster near-duplicates and apply per-cluster limits.

---

## 9.2 Make dedup split-aware

Prevent:
- the same semantic family flooding one split,
- the same paraphrase family dominating train and test,
- novel clusters leaking across splits.

Use limits such as:
- max rows per semantic cluster,
- max per cluster per split,
- max per cluster per domain family.

---

## 9.3 Make selection diversity-aware

When choosing among duplicate or near-duplicate rows, prefer rows that add:
- new domain coverage,
- new sentiment coverage,
- new ambiguity type coverage,
- new support-type coverage,
- new hard-case coverage.

---

## 10. Abstain redesign

## 10.1 Treat abstain as a benchmark state

Add explicit abstain reasons such as:
- `insufficient_evidence`
- `competing_interpretations`
- `weak_domain_signal`
- `novel_but_unstable`

This makes abstain rows analyzable rather than accidental.

---

## 10.2 Use calibrated abstain logic

Do not rely only on raw confidence.
Use a calibrated score based on:
- classifier score,
- evidence score,
- ambiguity score,
- novelty score,
- soft mismatch penalty.

Recommended V7 approach:
- post-hoc calibrated selective thresholding,
- risk-coverage reporting,
- optional conformal-style abstain calibration when validation supports it.

---

## 10.3 Add abstain-specific evaluation

Report:
- abstain precision/recall,
- coverage-risk curves,
- accepted-risk vs abstain-rate,
- abstain reason distributions,
- abstain by domain,
- abstain by hardness.

---

## 11. Novelty redesign

## 11.1 Keep novelty as a V2-style extension, but stabilize it

Primary novelty routing method should remain:
- **distance + calibrated threshold**

LLM-based novelty judgment should remain optional and secondary.

---

## 11.2 Represent novel outputs as cluster IDs first

Use:
- `novel_cluster_id`
- optional alias
- cluster evidence summary

Do not depend on free-text aspect names as the primary novelty representation.

---

## 11.3 Add novelty calibration prerequisites

Only run novelty calibration when validation includes:
- enough known positives,
- enough novel positives,
- enough cross-domain support.

Otherwise:
- report that calibration was skipped,
- fall back to conservative distance thresholds.

---

## 11.4 Strengthen novelty leakage audit

Audit:
- cluster-level leakage,
- alias-level leakage,
- support-text leakage,
- cluster saturation by split.

---

## 12. Evidence and support-type redesign

## 12.1 Expand support taxonomy

Suggested support types:
- `exact`
- `near_exact`
- `symptom_based`
- `paraphrastic`
- `contrastive`
- `coref_resolved`
- `domain_consistent_weak`
- `novel_support`

### Policy by target
- benchmark gold: mostly exact / near_exact / high-quality symptom_based
- train: allow most grounded weak types
- silver: allow all grounded types but annotate reliability clearly

---

## 12.2 Add support reliability score

Instead of only rejecting by support type, compute a reliability score based on:
- support prior,
- evidence overlap,
- clause precision,
- contradiction penalty,
- domain agreement.

Then use this score for ranking and promotion.

---

## 13. Domain and ontology redesign

## 13.1 Separate hard and soft domain mismatch

### Hard mismatch
- impossible aspect for domain,
- weak evidence,
- strong conflict.

### Soft mismatch
- semantically plausible,
- strong evidence,
- potentially useful cross-domain transfer signal.

Soft mismatch should often go to silver or train_keep instead of hard reject.

---

## 13.2 Stabilize the canonical implicit inventory

Use a moderate universal inventory for the implicit branch.

### Target size
- roughly **30–60 canonical implicit aspects**

This is enough for coverage but small enough for consistent evaluation.

---

## 13.3 Improve learned-ontology governance

Promotion of new aspects into the official ontology should require:
- stability across runs,
- cross-domain support,
- semantic distinctness,
- duplicate-alias suppression.

---

## 14. Synthetic and weak-supervision redesign

## 14.1 Use a three-stream data strategy

### Stream A — raw multi-domain reviews
Main diversity source.

### Stream B — public benchmark datasets
Use for sanity-check evaluation and controlled comparisons.

### Stream C — synthetic implicit data
Generate through:
- explicit→implicit paraphrases,
- symptom→aspect rules,
- controlled LLM rewrites,
- weakly supervised pseudo-labeling.

---

## 14.2 Make synthetic generation challenge-aware

Synthetic generation should target shortages in:
- telecom,
- positive and negative sentiment,
- H2/H3 hard cases,
- abstain-worthy ambiguity,
- novelty clusters.

It should also reject:
- near-duplicates,
- prompt-collapse outputs,
- low-diversity repeated templates.

Each synthetic row should retain:
- `generation_source`
- `generator_policy`
- evidence span
- domain family
- intended role (train / silver / stress-test)

---

## 14.3 Build an explicit↔implicit paraphrase bank

Store pairs with:
- explicit statement,
- implicit paraphrase,
- domain,
- aspect,
- sentiment,
- evidence span.

Use these for:
- weak supervision,
- augmentation,
- retrieval grounding,
- novelty stress tests.

---

## 15. Train export redesign

## 15.1 Use target bands, not one target

Set separate minimums for:
- total train rows,
- per-domain rows,
- per-sentiment rows,
- hard-case rows,
- max duplicate-cluster share.

---

## 15.2 Make top-up usefulness-aware

Top-up should prioritize rows that fill deficits in:
- telecom,
- positive/negative sentiment,
- novelty,
- abstain,
- H2/H3,
- low-duplicate clusters.

For train top-up only, `general` should no longer be an automatic rejection if the row is:
- grounded,
- domain-compatible,
- useful for current deficits,
- not duplicate-heavy.

---

## 15.3 Add a train sentiment rescue pass

Before final train export, explicitly search silver rows for:
- negative examples,
- positive examples,
- underrepresented domains,
- lower-duplicate alternatives.

This should happen before the final train artifact is written.

---

## 16. Performance plan

## 16.1 Cache every expensive operation

Persist caches for:
- LLM calls,
- embeddings,
- novelty cluster lookup,
- semantic dedup clusters,
- aspect registry lookups,
- evidence span normalization.

Cache keys should include:
- normalized text,
- domain,
- model version,
- rule version,
- prompt version.

---

## 16.2 Budget LLM fallback by row value

Only use LLM fallback for:
- high-usefulness borderline rows,
- rare domains,
- hard-case candidates,
- novelty candidates,
- abstain-or-answer cases where cheap methods disagree.

Do not call LLM uniformly.

---

## 16.3 Make chunked research runs representative

Support:
- research-profile chunked runs,
- resumable dedup index,
- resumable novelty index,
- partial benchmark rebuild,
- state reuse across chunks.

Goal:
small laptop runs should reflect the real research logic without collapsing into empty exports.

---

## 16.4 Add profiling sidecars

Each run should report:
- per-stage runtime,
- LLM call count,
- embedding count,
- cache hit rates,
- dedup cluster count,
- approximate memory usage.

---

## 17. Reporting and governance redesign

## 17.1 Improve human-readable summaries

Every run should show:
- benchmark gold count,
- benchmark silver count,
- train_keep count,
- hard_reject count,
- salvage recovered count,
- top-up accepted count,
- abstain rows,
- novel rows,
- duplicate cluster count,
- family floor usage,
- score distributions.

---

## 17.2 Make blockers explainable

Each blocker should say:
- what failed,
- where it failed,
- how many rows were affected,
- which module decided it,
- what the likely recovery action is.

---

## 17.3 Align promotion wording with run mode

If a run is `research_release`, reporting should not look debug-like unless a real blocker exists.

---

## 17.4 Add a “why not promoted?” sidecar

Whenever promotion fails, emit a compact diagnostic sidecar with:
- top failing metrics,
- top failing domains,
- top failing reasons,
- rows that narrowly missed promotion,
- suggested threshold or policy interventions.

---

## 18. Evaluation redesign

## 18.1 Separate benchmark quality from model quality

### Benchmark quality metrics
- grouped split leakage,
- domain-family coverage,
- semantic duplication,
- evidence grounding,
- implicit purity,
- ontology compatibility,
- abstain coverage,
- novelty coverage,
- hardness coverage.

### Model quality metrics
- flexible macro-F1,
- Jaccard overlap,
- aspect-only F1,
- selective risk/coverage,
- abstain precision/recall,
- novelty AUROC/F1,
- calibration error.

---

## 18.2 Add benchmark diversity metrics

Track:
- unique semantic cluster count,
- cluster entropy,
- per-domain diversity,
- sentiment diversity,
- ambiguity type diversity,
- support-type diversity.

---

## 18.3 Track worst-domain quality explicitly

In addition to worst-domain regression, track:
- worst-domain purity,
- worst-domain grounding,
- worst-domain abstain quality,
- worst-domain novelty false positives.

---

# Part II — ProtoNet in V7

## 19. ProtoNet’s role in V7

ProtoNet stays in V7, but **not as the center of the project**.

### ProtoNet should be:
- a strong few-shot baseline,
- an optional prototype head for the implicit branch,
- a distance-based scorer for abstain and novelty routing.

### ProtoNet should not be:
- the paper identity,
- the only implicit model,
- a pure closed-set joint-label-only system.

The benchmark and selective inference remain the main research identity.

---

## 20. Best placement of ProtoNet in the project

ProtoNet should have three roles.

### Role A — Main few-shot baseline
Use it as the clean baseline for implicit aspect inference.

### Role B — Prototype head option
Use it as an optional head on top of a stronger text encoder.

### Role C — Distance-based novelty/abstain scorer
Use prototype distance for:
- known-vs-novel routing,
- low-confidence abstain,
- ambiguity from top-2 closeness.

---

## 21. What should change in ProtoNet for V7

## 21.1 Move away from pure joint-label closed-set classification

Do not keep ProtoNet limited to:
- `aspect__sentiment` only.

V7 ProtoNet should support:
- aspect-only candidate scoring,
- evidence-conditioned scoring,
- selective answer / multi-label / abstain,
- known-vs-novel distance logic.

Joint aspect+sentiment should remain as one baseline setting, not the only task mode.

---

## 21.2 Make ProtoNet evidence-conditioned

ProtoNet inputs should be built from:
- evidence sentence or evidence span,
- review context only when needed,
- aspect candidate inventory,
- domain metadata when useful.

This is better than always using full review text.

---

## 21.3 Improve the encoder under ProtoNet

Do not spend most effort inventing a new prototype formula.
Focus on the encoder.

Recommended stack:
1. weakly supervised noisy ABSA pretraining,
2. explicit↔implicit paraphrase/contrastive learning,
3. few-shot adaptation with ProtoNet or prototype head.

This will help more than heavily modifying the prototype computation.

---

## 21.4 Add proper selective-inference evaluation for ProtoNet

Evaluate ProtoNet with:
- flexible match against multiple acceptable labels,
- multi-label overlap,
- abstention metrics,
- coverage-risk,
- domain/group holdout breakdown,
- novelty AUROC/F1 when novelty is enabled.

Do not evaluate it only with plain exact accuracy.

---

## 21.5 Add sanity checks for suspiciously high scores

Every ProtoNet run should automatically log:
- semantic duplicate rate,
- grouped split leakage,
- source/product leakage,
- domain-holdout score,
- calibration quality,
- nearest-neighbor label leakage audit.

This is necessary because few-shot metrics can look artificially high on overly easy or leaky splits.

---

## 22. ProtoNet architecture in V7

Suggested layout:

```text
implicit_models/
  protonet_baseline.py
  prototype_head.py
  encoder_backbones.py
  support_query_builder.py
  protonet_eval.py
  prototype_novelty.py
```

### Responsibilities
- `protonet_baseline.py`: classic few-shot baseline
- `prototype_head.py`: prototype scoring head for stronger encoders
- `encoder_backbones.py`: transformer/BOW/other encoders
- `support_query_builder.py`: benchmark-aware input shaping
- `protonet_eval.py`: flexible + selective evaluation
- `prototype_novelty.py`: novelty and abstain scoring from prototype distance

---

## 23. ProtoNet-specific formulas to keep

### Prototype for class c

\[
p_c = \frac{1}{|S_c|} \sum_{x_i \in S_c} f_\theta(x_i)
\]

### Candidate score

\[
s(q, c) = \cos(f_\theta(q), p_c)
\]

### Label probability

\[
P(c \mid q) = \frac{\exp(s(q,c)/\tau)}{\sum_{c'} \exp(s(q,c')/\tau)}
\]

### Ambiguity score

\[
A(q) = 1 - (p_1 - p_2)
\]

### Novelty score

\[
N(q) = 1 - \max_c \cos(f_\theta(q), p_c)
\]

These are still useful because they are simple, interpretable, and fit selective inference naturally.

---

## 24. How ProtoNet interacts with the V7 builder

The builder should produce artifacts that ProtoNet consumes.
ProtoNet should not shape builder policy directly.

### Builder responsibilities
- construct benchmark instances,
- produce train examples,
- preserve evidence,
- assign split protocols,
- track abstain and novelty metadata.

### ProtoNet responsibilities
- score candidates,
- support selective decisioning,
- support novelty routing,
- act as a credible few-shot baseline.

---

## 25. V7 implementation roadmap

## Phase 1 — Architecture cleanup

### Goals
- shrink orchestration,
- isolate policy modules,
- formalize row contracts,
- stabilize summaries.

### Tasks
1. split `build_dataset.py`
2. add row lifecycle state models
3. move scoring and bucketing into `curation/`
4. move benchmark export into `benchmark/`
5. move train export into `trainset/`
6. add profiling sidecars

### Exit criteria
- orchestration file <= 900 lines,
- policy modules unit-tested,
- regression snapshots preserved.

---

## Phase 2 — Fix over-rejection and train underfill

### Goals
- promote more useful silver rows,
- rescue sentiment diversity,
- preserve telecom and minority families.

### Tasks
1. add multi-objective row score
2. make top-up usefulness-aware
3. allow train-only weak support types
4. add sentiment rescue pass
5. re-run family floors after promotion

### Exit criteria
- top-up covers >= 20% of shortfall on small runs,
- train rows no longer collapse,
- telecom survives export consistently,
- positive/negative counts remain nonzero.

---

## Phase 3 — Add semantic dedup and diversity controls

### Goals
- lower duplicate logical row rate,
- increase benchmark diversity,
- prevent cluster flooding.

### Tasks
1. exact/fuzzy dedup
2. semantic dedup
3. per-cluster export limits
4. diversity-aware selection

### Exit criteria
- duplicate rate drops substantially,
- diversity metrics improve,
- novelty clusters remain split-clean.

---

## Phase 4 — Stabilize abstain and novelty

### Goals
- make abstain calibrated,
- make novelty auditable and stable.

### Tasks
1. add calibrated selective decision layer
2. add abstain reason taxonomy
3. add novelty calibration prerequisites
4. expand novelty leakage audits

### Exit criteria
- abstain metrics reported consistently,
- novelty calibration only runs when valid,
- no cross-split novelty leakage.

---

## Phase 5 — Improve synthetic and weak supervision

### Goals
- improve train coverage without polluting benchmark,
- boost rare domains and hard cases.

### Tasks
1. build paraphrase bank
2. add challenge-aware synthetic generator
3. add pseudo-label weak-supervision path
4. add dedup-aware synthetic acceptance

### Exit criteria
- train shortfall decreases,
- telecom/minority sentiment improve,
- synthetic rows do not inflate duplication excessively.

---

## Phase 6 — Governance and release polish

### Goals
- make failed runs interpretable,
- produce release-grade summaries.

### Tasks
1. align promotion labels with real blockers
2. add `why_not_promoted` sidecar
3. expose gold/silver/train/recovery counts in markdown summary
4. tighten release signoff rules

### Exit criteria
- report wording is consistent,
- failed runs explain themselves clearly,
- promoted runs are easy to justify.

---

## 26. Success criteria for V7

V7 is successful if it achieves:

### Architecture
- smaller orchestration file,
- separated policy modules,
- explicit row lifecycle.

### Data quality
- lower duplicate logical row rate,
- better implicit purity,
- high ontology compatibility retained.

### Train usability
- less chronic underfill,
- better telecom retention,
- better positive/negative retention,
- top-up materially contributing.

### Benchmark behavior
- grouped split leakage remains zero,
- abstain and novelty remain available,
- family floors preserved consistently.

### ProtoNet role
- remains a clean baseline,
- supports abstain/novelty scoring,
- evaluated under benchmark-aware selective metrics,
- does not become the paper identity.

### Performance
- high cache reuse,
- faster chunked research runs,
- budgeted LLM fallback,
- per-stage profiling available.

---

## 27. Final recommendation

V7 should not be “V6 plus more heuristics.”
It should be a **structural cleanup plus a data-centric redesign**.

The central strategy is:

> **Keep gold benchmark rows strict, make train recovery usefulness-aware, add semantic deduplication, calibrate abstain and novelty, and keep ProtoNet as a strong baseline/component rather than the paper identity.**

If executed well, V7 should become:
- easier to maintain,
- easier to trust,
- more scientifically defensible,
- and more practical for both small laptop runs and larger research-profile releases.
