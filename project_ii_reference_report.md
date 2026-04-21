# Project II / ReviewOps — Reference Report

## 1. Project identity

**Project name:** Project II / ReviewOps  
**Current focus:** Review-based, ambiguity-aware, evidence-grounded aspect understanding for user reviews  
**Application type:** Research project plus deployable review intelligence system  
**Primary input:** Natural-language reviews from multiple domains such as restaurants, electronics, telecom, services, and related review settings  
**Primary output:** Explicit aspects, implicit aspects, aspect-level sentiment, grounded evidence spans, aggregated analytics, and downstream review intelligence

This project is **not** a fixed-category sentiment classifier and **not** only a document-level sentiment model. Its goal is to understand **why** a review is positive or negative by identifying the specific aspects being discussed, including aspects that are stated directly and aspects that are implied indirectly. The system is designed to support both **research evaluation** and **operational analytics**. fileciteturn3file0L4-L12

---

## 2. What the project tries to solve

Most sentiment systems reduce a review to one overall label such as positive, neutral, or negative. That loses the fine-grained reasons behind user satisfaction or dissatisfaction. In practical settings, organizations need to know whether problems come from delivery, support, connectivity, staff behavior, battery life, cleanliness, price, or other recurring issues.

Project II is intended to solve the following problems:

1. **Open-aspect review understanding**  
   The system should discover aspects from the review text itself rather than force all reviews into a small predefined aspect taxonomy.

2. **Implicit aspect detection**  
   Reviews often express complaints or praise without naming the target explicitly. For example, “I had to call three times before it connected” implies a network/connectivity issue even if the word “network” is absent.

3. **Evidence-grounded reasoning**  
   Every prediction should be tied to evidence from the review so the output is inspectable, explainable, and usable in research or analytics.

4. **Ambiguity-aware interpretation**  
   A review may support more than one valid interpretation. The project therefore treats ambiguity and abstention as part of the benchmark design rather than as annotation noise.

5. **Cross-domain review intelligence**  
   The system is meant to work beyond one narrow benchmark domain and later support dashboards, alerts, summaries, and operational monitoring. fileciteturn3file0L14-L31

---

## 3. High-level aim

The aim of the project is to build a **domain-agnostic, evidence-first, hybrid aspect-based sentiment analysis system** that:

- extracts explicit aspects directly mentioned in a review,
- infers implicit aspects from contextual cues,
- assigns sentiment at the aspect level,
- preserves grounded evidence spans,
- supports novelty and ambiguity-aware evaluation,
- and enables downstream analytics through ReviewOps.

In short, the project tries to bridge the gap between **raw review text** and **actionable, aspect-level review intelligence**.

---

## 4. Core idea of the current methodology

The **latest active methodology** is a **hybrid pipeline** with three major reasoning layers:

1. **Explicit aspect detection**  
   Finds directly mentioned aspect phrases from the review.

2. **Implicit aspect detection / inference**  
   Infers hidden aspects that are not explicitly named but are supported by symptoms or contextual evidence.

3. **LLM validation / reasoning layer**  
   Validates, refines, merges, removes, or adds aspect interpretations only when the review evidence supports them.

This means the project is no longer framed as a pure MAML system or a pure graph system. The **current official direction is hybrid**: explicit extraction + implicit inference + evidence-grounded validation, with sentiment and graph/analytics as downstream components.

This is important because some earlier project directions emphasized graph-regularized open-aspect extraction as the main story. That older framing still matters historically, but the **latest reference methodology** is the hybrid evidence-grounded approach above. fileciteturn3file3L1-L18

---

## 5. Problem framing in research terms

Scientifically, the project sits at the intersection of:

- open-aspect extraction,
- implicit aspect inference,
- aspect-based sentiment analysis,
- evidence-grounded interpretation,
- selective prediction / abstention,
- ambiguity-aware benchmarking,
- and review analytics.

The project differs from classic ABSA pipelines in two main ways:

- It does **not** assume that all aspects come from a fixed, closed set.
- It treats **implicit and ambiguous interpretations** as first-class parts of the task.

That makes the project closer to a **review intelligence system** than a standard benchmark-only ABSA classifier.

---

## 6. Latest end-to-end methodology

## 6.1 Stage A — Explicit aspect extraction

The explicit branch is designed for **open-domain explicit aspect detection**.

### Main idea
The system extracts aspect candidates directly from the review text instead of choosing from a fixed ontology at inference time.

### Current method
The current explicit extractor is syntax-guided and evidence-oriented. It uses:

- spaCy tokenization and linguistic analysis,
- noun chunks,
- dependency-based rules,
- filtering of generic or low-value phrases,
- embedding-based de-duplication,
- and lightweight canonicalization.

### Behavior
Typical extraction includes:

- direct noun phrases,
- aspect heads supported by modifiers,
- cleaned multi-word aspect expressions,
- removal of generic fragments,
- keeping longer, more informative phrases when duplicates occur.

This produces a list of **explicit aspect candidates** such as:

- `battery life`
- `delivery`
- `service quality`
- `call quality`

Earlier implementation summaries describe the open extractor as spaCy noun chunking plus filtering, substring cleanup, and embedding de-duplication using MiniLM-style sentence embeddings. fileciteturn3file1L16-L37

### Why this branch exists
This branch gives the pipeline:

- interpretability,
- fast explicit coverage,
- open-domain flexibility,
- and stable evidence linking.

It is the deterministic foundation of the hybrid system.

---

## 6.2 Stage B — Implicit aspect inference

The implicit branch is used when the aspect is **not named directly** but is still inferable from the review.

### Main idea
Given review evidence such as symptoms, outcomes, or behavior descriptions, infer the hidden aspect.

Example:
- “I had to try calling three times before it went through” → likely `network` or `connectivity`
- “It was at 20% by late afternoon” → likely `battery life`

### Current position
The project no longer treats pure MAML as the main solution. The recommended latest direction is:

- **strong text encoder** for implicit reasoning,
- **weakly supervised noisy ABSA pretraining**,
- **explicit↔implicit paraphrase/contrastive learning**,
- and optionally a **few-shot head** such as ProtoNet or ANIL on top.

This is the active methodological view because pure MAML was considered too brittle and too weak as the headline method for this review task. fileciteturn3file3L20-L54 fileciteturn3file4L20-L54

### Canonical implicit aspect space
The implicit branch should not be fully open in the same way as the explicit branch. Instead, it should map symptoms into a **moderate canonical aspect inventory**, for example:

- price / value
- delivery / shipping
- customer support
- staff behavior
- service speed
- battery life
- charging
- connectivity / network
- call quality
- display
- comfort
- cleanliness
- food quality
- ambience

This stabilizes annotation, evaluation, and model behavior. fileciteturn3file2L40-L82

---

## 6.3 Stage C — Evidence-grounded LLM verifier

After explicit and implicit candidate aspects are generated, a validation layer checks them against the review.

### Main idea
The verifier should not hallucinate free-form labels. It should make structured decisions only when evidence supports them.

### Intended verifier actions
For each candidate aspect, the verifier can:

- keep,
- drop,
- merge into another aspect,
- add an aspect if strong evidence supports it,
- and attach a confidence or judgment rationale.

### Constraint
All decisions must remain **evidence-grounded**. If the review does not support a prediction, the system should prefer dropping or abstaining instead of inventing an interpretation.

### Why this matters
This layer is central to the project’s current novelty direction because it:

- reduces noisy aspect candidates,
- prevents unsupported hidden-aspect additions,
- improves interpretability,
- and fits the ambiguity-aware benchmark design.

---

## 6.4 Stage D — Aspect-level sentiment classification

Once final aspects are decided, the system predicts sentiment for each aspect.

### Current sentiment direction
The current project memory indicates an **aspect-conditioned seq2seq classifier** that scores labels such as:

- positive
- neutral
- negative

The earlier MVP implementation used a prompt-based FLAN-T5 style model that takes the review plus a target aspect and returns one sentiment label. The output is constrained to the valid label set to avoid hallucinated formats. fileciteturn3file1L40-L61

### Scientific role
This stage turns the system from aspect discovery into full **aspect-based sentiment analysis**.

---

## 6.5 Stage E — Evidence extraction

For every aspect prediction, the system preserves supporting evidence from the original review.

### Stored evidence information
Typical stored fields include:

- aspect phrase,
- evidence text,
- character span,
- confidence,
- and sentiment.

This supports explainability, auditing, annotation inspection, and analytics. Evidence-first design is one of the main project principles. fileciteturn3file0L64-L75

---

## 6.6 Stage F — Graph layer and downstream analytics

The graph component remains part of the project, but in the latest framing it is **downstream and supportive**, not the sole core novelty.

### Main graph idea
Build an aspect graph where:

- nodes are aspect phrases or canonical aspect clusters,
- edges represent similarity, co-occurrence, or relatedness,
- and graph statistics are used for clustering, aggregation, and analytics.

### What the graph is for
The graph helps with:

1. **normalizing aspect variants**  
   e.g. `battery backup` ≈ `battery life`

2. **clustering related aspect phrases**

3. **stabilizing sentiment at aggregate level**

4. **detecting emerging issues or aspect clusters**

5. **powering analytics and dashboard grouping**

Earlier project context describes similarity or co-occurrence edges, optional PMI, and graph-based sentiment smoothing as part of the broader ReviewOps design. fileciteturn3file0L39-L62

### Current role of the graph
In the latest project direction, the graph is best treated as:

- cross-review normalization,
- aggregation and analytics infrastructure,
- possibly regularization or smoothing,
- and downstream operational intelligence support.

It is **not** the only headline contribution anymore; it complements the hybrid explicit–implicit–verifier pipeline.

---

## 7. Dataset builder

The **dataset_builder** is one of the most important parts of Project II because the project needs a benchmark that existing public datasets do not fully provide.

## 7.1 Why dataset_builder exists

Public ABSA datasets such as SemEval-2014 and MAMS are useful, but they are not enough for the project’s goal because:

- they are relatively domain-limited,
- they focus more on classic explicit ABSA tasks,
- they do not fully represent implicit open-world review reasoning,
- and they do not naturally encode ambiguity, abstention, and novelty the way this project needs. fileciteturn3file2L1-L39

So dataset_builder is used to create a custom benchmark and training pipeline for the project.

---

## 7.2 Current goals of dataset_builder

The dataset builder is intended to create data that supports:

- explicit and implicit aspect supervision,
- evidence-grounded labels,
- ambiguity-aware gold annotations,
- abstention-acceptable cases,
- novelty / unseen aspect evaluation,
- grouped split protocols to reduce leakage,
- and multi-domain review coverage.

This is not just a file converter. It is a **benchmark-construction pipeline**.

---

## 7.3 What the latest benchmark direction looks like

From the recent V7 work, the builder is producing structured benchmark rows with fields and signals such as:

- review text,
- domain and domain family,
- group id,
- gold interpretations,
- explicit grounded interpretations,
- implicit grounded interpretations,
- novelty flags,
- abstain flags,
- ambiguity score,
- hardness tier,
- and evidence grounding.

This means the benchmark is designed for more than simple classification. It supports:

- ambiguity-aware evaluation,
- selective inference,
- novelty analysis,
- explicit vs implicit reasoning studies,
- and later verifier or graph-layer evaluation.

---

## 7.4 Current state of the latest benchmark work

The recent V7 artifact work shows the builder is moving in the right direction, but still has unresolved quality issues.

### What is already strong
- structured research artifact packaging,
- explicit/implicit split in the labels,
- evidence-grounded rows,
- ambiguity-heavy challenge slices,
- novelty-aware benchmark design,
- and grouped split discipline.

### What is still weak
- benchmark slices are often too small,
- logical/semantic duplication remains a risk,
- implicit purity is still weaker than desired,
- ontology/canonicalization is not fully stable,
- and some runs have suffered from domain coverage imbalance.

So the latest conclusion is:

> dataset_builder is becoming research-grade structurally, but benchmark scale and implicit-label quality remain the main bottlenecks.

---

## 7.5 Current data strategy for the project

The strongest current recommendation is a **3-layer data strategy**:

1. **Large multi-domain raw review corpora** for coverage  
   e.g. Amazon Reviews 2023 and Yelp-style review data

2. **Public ABSA gold datasets only for evaluation and sanity-check benchmarking**  
   e.g. SemEval-2014 and MAMS

3. **A project-specific synthetic implicit dataset** built from real review language, explicit→implicit transformations, paraphrase pairs, and weak supervision

This is the most suitable data plan for the current methodology. fileciteturn3file2L9-L38 fileciteturn3file2L84-L140

---

## 8. ProtoNet and meta-learning status

ProtoNet has an important role in the project history, but its role has changed.

## 8.1 Why ProtoNet was considered

The project needed a way to perform few-shot implicit aspect inference when there are limited explicit gold labels for hidden-aspect reasoning. ProtoNet was attractive because it:

- is simpler than full MAML,
- works with prototype representations,
- can adapt to label-limited settings,
- and can be attached as a lightweight head over a strong encoder.

---

## 8.2 Current official role of ProtoNet

ProtoNet is **not the project identity**.

The latest reference direction treats ProtoNet as one of the following:

- a baseline,
- a component of the implicit branch,
- or a lightweight few-shot head over a stronger pretrained encoder.

It is **not** the headline novelty.

The project’s current recommended story is:

- explicit extraction + implicit inference + verifier as the main method,
- with ProtoNet, ANIL, or similar approaches used as baselines or subcomponents.

This is because pure few-shot meta-learning alone is not considered sufficient for the full task. fileciteturn3file3L55-L104 fileciteturn3file4L55-L104

---

## 8.3 Why pure MAML is not the current main method

The latest project guidance explicitly recommends **not** using vanilla MAML as the main headline method because:

- it is too brittle,
- it can overfit or memorize small support sets,
- it is not well matched to the real structure of review-based implicit inference,
- and more recent ABSA work suggests stronger encoder learning plus weak supervision and paraphrase learning are more suitable.

So the up-to-date project position is:

- **do not present pure MAML as the final project methodology**,
- but **do keep ProtoNet / MAML / ANIL as baselines or optional components in the implicit branch**.

---

## 8.4 Note on suspiciously high ProtoNet results

The project memory also includes an important caution: any extremely high ProtoNet result, such as near-perfect few-shot accuracy, should be treated with skepticism until leakage checks are done. Potential causes include:

- duplicate leakage,
- paraphrase leakage,
- aspect-word leakage,
- poor grouped split discipline,
- or over-easy evaluation settings. fileciteturn3file3L70-L94 fileciteturn3file4L70-L94

That warning remains part of the project reference.

---

## 9. Current system architecture view

At the current stage, the best high-level architecture for the project is:

```text
Review
  ↓
Explicit aspect extraction
  ↓
Implicit aspect inference
  ↓
Evidence-grounded verifier
  ↓
Aspect-level sentiment classification
  ↓
Evidence span attachment
  ↓
Aspect clustering / graph aggregation
  ↓
Storage + analytics + alerts + summaries
```

This combines the older open-aspect ReviewOps pipeline with the newer hybrid explicit/implicit/verifier methodology.

---

## 10. What has already been built or defined

Based on the latest project state, the following are part of the current system definition or implemented direction:

### Already defined clearly
- open explicit aspect extraction,
- aspect-conditioned sentiment prediction,
- evidence extraction,
- graph-based downstream aggregation idea,
- dataset_builder benchmark pipeline,
- novelty and ambiguity-aware evaluation direction,
- ReviewOps analytics concept,
- and hybrid methodology positioning.

### Already built in earlier MVP form
- FastAPI backend,
- MySQL persistence,
- review inference endpoint,
- CSV ingestion endpoint,
- open-aspect extraction pipeline,
- seq2seq sentiment classification,
- evidence span storage. fileciteturn3file1L93-L152

### Under active research refinement
- implicit branch training pipeline,
- dataset_builder V7 quality,
- canonical aspect inventory,
- verifier prompting and schema,
- graph regularization/aggregation details,
- final benchmark release readiness.

---

## 11. Intended operational ReviewOps capabilities

Project II is not only a model pipeline. It is also intended to support a ReviewOps-style intelligence system.

### Intended downstream capabilities
- aspect frequency counts,
- per-aspect sentiment distribution,
- aspect sentiment scores,
- top negative aspects,
- trend analysis over time,
- alerts for drops or emerging issues,
- drill-down views with evidence,
- evidence-grounded weekly summaries,
- and optional ticketing or action-center flows. fileciteturn3file0L117-L139

So the project has both a **research track** and an **operational analytics track**.

---

## 12. Evaluation philosophy

The project’s evaluation is broader than conventional ABSA metrics.

### Model-level metrics
- aspect extraction quality,
- sentiment accuracy / macro-F1,
- evidence grounding quality,
- implicit inference quality,
- novelty performance,
- and abstention/selective prediction behavior.

### Benchmark-level quality checks
- leakage control,
- grouped split quality,
- duplication rate,
- ontology compatibility,
- implicit purity,
- domain coverage,
- and ambiguity density.

### System-level metrics
- inference latency,
- dashboard responsiveness,
- aggregation quality,
- and operational usefulness.

Earlier ReviewOps context also listed metrics such as span F1, macro-F1, graph contribution metrics, faithfulness score, and dashboard load time. fileciteturn3file0L97-L111

---

## 13. Current strengths of the project

1. **Clear research identity**  
   The project now has a defensible hybrid methodology instead of relying on one fragile component.

2. **Evidence-first design**  
   Predictions are intended to remain grounded in review spans.

3. **Open + implicit coverage**  
   The system handles both named and hidden aspects.

4. **Benchmark ambition beyond classic ABSA**  
   The benchmark work includes ambiguity, abstention, and novelty.

5. **Practical deployment path**  
   The project can support dashboards, alerts, summaries, and operational review intelligence.

6. **Good modularity**  
   Explicit extraction, implicit inference, verification, sentiment, graph aggregation, and analytics are conceptually separable.

---

## 14. Current weaknesses or unresolved issues

1. **Final benchmark quality is not fully mature yet**  
   The dataset_builder artifacts have improved structurally, but clean large-scale release quality is not yet fully achieved.

2. **Implicit purity remains a bottleneck**  
   The implicit branch and benchmark still need cleaner supervision.

3. **Canonical aspect normalization is incomplete**  
   Some aspect labels remain too fragmented or too generic.

4. **Semantic duplication risk remains**  
   Even when exact text duplication is controlled, semantic repetition can still inflate performance.

5. **Graph role must be framed carefully**  
   It should be presented as a supporting aggregation and analytics strength, not an outdated sole identity if that no longer matches the latest methodology.

6. **Meta-learning results must be interpreted cautiously**  
   Very high ProtoNet-style results are not automatically trustworthy without strong leakage checks.

---

## 15. What the project is not

To avoid confusion, the current project should **not** be described as any of the following:

- purely a vanilla MAML paper,
- purely a graph paper,
- purely a fixed-category ABSA classifier,
- purely a seq2seq sentiment model,
- or only a dashboard application.

It is a **hybrid review intelligence project** that combines modeling, benchmark construction, evidence-grounded reasoning, and analytics.

---

## 16. Current recommended one-paragraph description

Project II is a hybrid, evidence-grounded review intelligence system designed to perform fine-grained understanding of user reviews across domains. It combines open explicit aspect extraction, implicit aspect inference, aspect-level sentiment classification, evidence span preservation, and downstream aspect aggregation for analytics. The project’s benchmark work is ambiguity-aware and novelty-aware, allowing multiple grounded interpretations, abstention-acceptable cases, and explicit separation of direct and hidden aspect reasoning. ProtoNet and related few-shot methods are treated as components or baselines for the implicit branch rather than the entire project identity, while graph-based aggregation supports normalization, clustering, and operational ReviewOps analytics.

---

## 17. Current recommended short description

Project II / ReviewOps is an ambiguity-aware, evidence-grounded hybrid ABSA system for review intelligence. It extracts explicit aspects, infers implicit aspects, assigns sentiment, preserves evidence, and supports downstream analytics through aspect clustering and graph-based aggregation.

---

## 18. Reference status summary

### Current official methodology
**Explicit extraction + implicit inference + evidence-grounded verifier + aspect sentiment + graph/analytics downstream**

### Current benchmark status
**V7 diagnostic benchmark work is structurally strong but still needs better scale, cleaner implicit labels, and stronger release-quality discipline**

### Current ProtoNet status
**Useful component/baseline for implicit few-shot inference, not the project headline**

### Current graph status
**Important downstream normalization and analytics layer, not the only project identity**

### Current project identity
**A review intelligence system and research benchmark effort, not only a classifier**

