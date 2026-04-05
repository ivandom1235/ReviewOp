# ReviewOp: Reasoning-Augmented Symbolic-Neural Synthesis for Domain-Agnostic Implicit Aspect Detection

## Abstract
Implicit Aspect Detection (IAD) remains a bottleneck in Aspect-Based Sentiment Analysis (ABSA) because implicit cues rarely contain direct aspect surface forms and vary strongly across domains. ReviewOp (v5.5) addresses this through a Symbolic-Neural Synthesis (SNS) design: deterministic latent-aspect grounding for reliability, plus optional LLM-assisted reasoned recovery for hard implicit cases. The resulting corpus is consumed by a few-shot Prototypical Network (ProtoNet) that learns joint aspect-sentiment prototypes for domain-robust inference.

---

## 1. Introduction: The Implicit Gap
Traditional ABSA pipelines are effective for explicit mentions but underperform on implicit evidence where aspect tokens are omitted. End-to-end neural systems can improve recall but often sacrifice transparency and controllability. ReviewOp enforces a division of labor: neural components transform language (paraphrase/rewrite), while symbolic modules retain final aspect assignment and span-grounding responsibility.

---

## 2. Methodology: Dataset Builder (Symbolic-Neural Synthesis)

Let the raw corpus be
\[
\mathcal{D}=\{(x_i,d_i,\ell_i)\}_{i=1}^N,
\]
where \(x_i\) is review text, \(d_i\) is canonical domain, and \(\ell_i\) is detected language.

### 2.1 Input Ingestion and Schema Discovery
The dataset builder (`dataset_builder/code/build_dataset.py`) ingests `csv/tsv/json/jsonl/xlsx/xml` files, preserves source metadata, and applies schema profiling (`schema_detect.py`) to identify:
1. primary text column,
2. numeric/categorical/datetime feature columns,
3. optional target-like column,
4. stable schema fingerprint.

Each row is assigned a deterministic ID and canonicalized domain from source filename.

### 2.2 Preprocessing and Eligibility Gating
Rows are transformed into a unified intermediate table with:
1. language tags (`language_utils.detect_language`),
2. implicit-readiness flag (`is_implicit_ready`) based on minimum token count and supported languages,
3. optional heuristic coreference rewriting (`coref.py`),
4. optional low-text drop (`min_text_tokens`, unless `no_drop`).

Data are split into train/val/test using stratified splitting when feasible (`splitter.py`).

### 2.3 Candidate Aspect Discovery
Latent aspect candidates are extracted from train only (`discover_aspects`) at three scopes:
1. global,
2. language-conditioned,
3. domain-conditioned.

The latent rulebook is fixed and includes categories such as `value`, `power`, `connectivity`, `performance`, `display quality`, `service quality`, and `food quality`.

### 2.4 Explicit Feature Branch
In parallel with implicit extraction, explicit artifacts are generated (`explicit_features.py`):
1. numeric values + MinMax normalization,
2. categorical IDs + one-hot encodings,
3. datetime decomposition (year/month/day-of-week/weekend),
4. text statistics (length, punctuation, capitalization, token count, etc.).

These are exported as the explicit track.

### 2.5 Implicit Branch: Clause-Level Symbolic-Neural Labeling
For each review, clause segmentation produces
\[
\mathcal{C}(x_i)=\{c_{i1},\dots,c_{im_i}\}.
\]
Each clause is sentiment-scored lexically and matched against latent rules. A match emits
\[
r=(a, s, t, h, \gamma),
\]
where \(a\) is latent aspect, \(s\) surface evidence, \(t\) support type, \(h\in\{0,1,2,3\}\) hardness, and \(\gamma\in[0,1]\) confidence.

Implementation hardness mapping:
1. `H0`: explicit lexical evidence (`label_type=explicit`, confidence 1.0),
2. `H1`: implicit lexical signal (`label_type=implicit`, confidence 0.85),
3. `H2`: LLM-parse/reasoned-recovery evidence,
4. `H3`: adversarially refined hard-implicit rewrite.

Aspect confidence is aggregated by max score:
\[
C_i(a)=\max_{r\in R_i(a)} \gamma_r.
\]
If no valid non-general label survives strict checks, the row falls back to `aspects=["general"]`.

### 2.6 Leakage and Grounding Constraints
Each match is filtered via strict leakage predicates (explicit contamination, latent name leakage, explicit keyword leakage, surface-equals-latent). Grounded spans are retained with support tags (`exact`, `near_exact`, `gold`, `llm_parse`, `reasoned_recovery`).

Strict gating is equivalent to:
\[
\mathbf{1}_{\text{strict}}(i)=
\mathbf{1}[A_i\neq\{\text{general}\}]\cdot
\mathbf{1}[|S_i|>0]\cdot
\mathbf{1}[\text{needs\_review}_i=0]\cdot
\mathbf{1}[q_i=\text{strict\_pass}].
\]

### 2.7 Reasoned Recovery (LLM-Assisted)
When symbolic matching is insufficient and recovery is enabled, clause paraphrasing is requested from an async provider (`RunPod`, `OpenAI`, `Ollama`, `Mock`) via `llm_utils.py`.

The recovery pipeline can be expressed as:
\[
\tilde{c}=\Phi_{\text{LLM}}(c,\mathcal{A}_{\text{cand}}),
\quad
\hat{a}=\Psi_{\text{rules}}(\tilde{c}).
\]

`\Phi_{LLM}` transforms text only; `\Psi_{rules}` remains the final aspect decision function. Responses are cached in a persistent hash-keyed cache (`.llm_cache.json`).

### 2.8 Async Execution and Train-Time Recovery Policies
Split construction is executed asynchronously (`asyncio.gather`) in bounded chunks. After initial build, train rows pass policy stages:
1. fallback-general capping,
2. strict leakage filtering,
3. re-inference for recoverable rows,
4. salvage/top-up recovery,
5. sentiment balancing,
6. train-size target enforcement,
7. strict subset derivation.

Quality metrics include:
\[
\text{FallbackRate}=\frac{1}{N}\sum_i\mathbf{1}[A_i=\{\text{general}\}],
\]
\[
\text{LeakageRate}=\frac{1}{N}\sum_i\mathbf{1}[\text{domain-leakage}(i)],
\]
\[
\text{GroundedRate}=\frac{\#\{i:A_i\neq\{\text{general}\},|S_i|>0\}}{\#\{i:A_i\neq\{\text{general}\}\}}.
\]

### 2.9 Output Artifacts
The pipeline exports:
1. `output/explicit/*.jsonl`,
2. `output/implicit/*.jsonl`,
3. `output/implicit_strict/*.jsonl`,
4. reports (`build_report.json`, `data_quality_report.json`),
5. ProtoNet compatibility datasets:
   - `output/compat/protonet/reviewlevel/*.jsonl`,
   - `output/compat/protonet/episodic/*.jsonl`.

---

## 3. Methodology: ProtoNet Few-Shot Learner

Let each processed example be
\[
e=(x,\tilde{x},d,a,s,w),
\]
with review text \(x\), evidence snippet \(\tilde{x}\), domain \(d\), aspect \(a\), sentiment \(s\), and confidence weight \(w\). The joint label is
\[
y=a\ \Vert\ \text{"__"}\ \Vert\ s.
\]

### 3.1 Data Interface and Adaptation
ProtoNet accepts:
1. `reviewlevel` rows with label lists,
2. `episodic` rows (examples or prebuilt episodes).

`reviewlevel_adapter.py` converts review-level rows to per-label example records with `example_id`, `parent_review_id`, `evidence_sentence`, confidence, and split metadata.

### 3.2 Episode Construction
If prebuilt episodes are absent, `episode_builder.py` constructs episodes under defaults:
1. `n_way=3`,
2. `k_shot=2`,
3. `q_query=2`.

Support/query overlap at parent-review level is forbidden to avoid leakage. Episode generation is deterministic via stable hashing + cached JSONL episode sets.

### 3.3 Encoder and Evidence-Aware Representation
`HybridTextEncoder` supports:
1. transformer backend (default `microsoft/deberta-v3-base`),
2. hashed BoW fallback when transformer loading is unavailable.

Input string format marks evidence span with `[E_START] ... [E_END]` and prefixes domain context (`[DOMAIN=...]`).

For transformer outputs \(H\in\mathbb{R}^{T\times d_h}\), with pooling mask \(m\in\{0,1\}^T\):
\[
z=\frac{\sum_{t=1}^{T}m_tH_t}{\sum_{t=1}^{T}m_t}.
\]
The projection head maps to normalized embedding
\[
u=\frac{g_\theta(z)}{\|g_\theta(z)\|_2}\in\mathbb{R}^{d_p},\quad d_p=256\ \text{(default)}.
\]

### 3.4 Prototype Computation and Metric Classification
For class \(k\), support embeddings \(\{u_j\}_{j\in\mathcal{S}_k}\), and confidence weights \(w_j\):
\[
p_k=\frac{\sum_{j\in\mathcal{S}_k}w_ju_j}{\sum_{j\in\mathcal{S}_k}w_j}.
\]
With smoothing \(\alpha\) and global mean \(\bar{p}\):
\[
p'_k=(1-\alpha)p_k+\alpha\bar{p},\quad \hat{p}_k=\frac{p'_k}{\|p'_k\|_2}.
\]
For query embedding \(u_q\), logits are
\[
\ell_k(u_q)=-\frac{\|u_q-\hat{p}_k\|_2^2}{T},
\]
where \(T=\exp(\tau)\) is learned temperature (clamped in implementation).

### 3.5 Training Objective
`trainer.py` minimizes:
\[
\mathcal{L}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{supcon},
\]
where \(\lambda=\) `contrastive_weight`.

Episode cross-entropy:
\[
\mathcal{L}_{CE}=-\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\log\frac{e^{\ell_{y_q}(u_q)}}{\sum_{k}e^{\ell_k(u_q)}}.
\]

Optimization stack:
1. AdamW with parameter-group LRs (projection vs encoder),
2. optional warmup contrastive epochs,
3. gradient accumulation,
4. CUDA AMP + GradScaler,
5. early stopping on validation accuracy.

### 3.6 Evaluation and Calibration
`evaluator.py` reports:
1. joint-label accuracy,
2. macro-F1,
3. aspect-only accuracy,
4. per-aspect accuracy,
5. low-confidence rate,
6. expected calibration error (ECE).

ECE is computed as:
\[
\text{ECE}=\sum_{b=1}^{B}\frac{|I_b|}{n}\left|\text{acc}(I_b)-\text{conf}(I_b)\right|,
\]
where \(I_b\) is confidence bin \(b\).

### 3.7 Export and Runtime Inference
After training, a global prototype bank is built from unique training examples with confidence weighting and smoothing (`prototype_bank.py`). The exported runtime bundle (`metadata/model_bundle.pt`) includes config, encoder/projection states, learned temperature, and prototype bank.

At inference (`runtime_infer.py`, `infer_api.py`), reviews are clause-split, scored against global prototypes, merged by aspect key, then truncated to top-\(K\). Neutral sentiment can be refined using Seq2Seq sentiment inference for final API outputs.

---

## 4. Engineering Notes (V5.5)
Implementation highlights include asynchronous LLM orchestration (`httpx` + `asyncio.gather`), persistent LLM caching, strict domain leakage controls, and direct compatibility between dataset exports and ProtoNet inputs.

---

## 5. Evaluation Protocol
The experimental workflow relies on:
1. internal build diagnostics (`build_report.json`) for fallback/leakage/grounding/sentiment constraints,
2. optional gold-label evaluation (`gold_eval`) for aspect/sentiment/span-overlap F1,
3. `run_experiment.py` sweeps + ablations with explicit quality gates and novelty gates.

---

## 6. Conclusion
ReviewOp establishes a practical research framework in which symbolic grounding remains the reliability core, while neural modules supply targeted linguistic recovery. Coupled with episodic ProtoNet learning, the system yields interpretable, controllable, and transferable implicit ABSA behavior across heterogeneous domains.
