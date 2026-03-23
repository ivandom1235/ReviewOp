# 🧠 ReviewOp Dataset Builder: Hybrid ABSA Pipeline

A domain-agnostic, multi-stage pipeline designed to transform raw consumer reviews into high-quality training data for **Aspect-Based Sentiment Analysis (ABSA)**. This system treats **Explicit Aspects** as open entities and **Implicit Aspects** as a canonical latent inventory.

---

## 🚀 Key Features

- **Hybrid Extraction:** Combines rule-based spaCy dependency parsing with LLM validation for explicit terms.
- **Weak Supervision:** Bootstraps implicit aspects (symptoms) using fuzzy matching (`RapidFuzz`) and semantic similarity (`all-mpnet-base-v2`).
- **Universal Taxonomy:** Grounded in 12 "Universal Superclasses" (Product Quality, Performance, Reliability, etc.) for cross-domain stability.
- **Data Augmentation:** Automated paraphrase diversification and cross-domain register transfer via LLM rewriting.
- **Dual Output Formats:**
  - **Seq2Seq (Format A):** Review-level strings optimized for generative models.
  - **ProtoNet (Format B):** Episodic N-Way K-Shot data for few-shot meta-learning.
- **Active Learning Loop:** Automated disagreement detection between proxy models with a static HTML UI (`review_interface.html`) for human-in-the-loop review.

---

## 🛠️ Pipeline Architecture (10 Stages)

1.  **Ingestion & Normalization:** Schema detection for CSV/Parquet/JSON and PII removal.
2.  **Data Source Strategy:** Domain mixing with Type A (Target), Type B (Open), and Type C (Gold) weighting.
3.  **Implicit Inventory:** Canonical symptom-to-aspect mapping using a dynamic BERTopic update utility.
4.  **Explicit Aspect Extraction:** Noun-chunk and opinion-word adjacency extraction with LLM filtering.
5.  **Implicit Aspect Labeling:** Semantic weak supervision using a pre-defined symptom library.
6.  **LLM Augmentation:** Hard negative generation and cross-domain transfer routines.
7.  **Evidence Span Auxiliary Model:** DeBERTa-base sentence classifier for precise span attribution.
8.  **Output Formatting:** Rigid sequence formatting and ProtoNet episode caching.
9.  **Splitting Strategy:** Temporal-bounded splitting with strict synthetic-data quarantine (Force-Train).
10. **Active Learning:** Uncertainty sampling based on prediction disagreements.

---

## 📁 Project Structure

```text
dataset_builder/
├── code/
│   ├── build_dataset.py      # Main pipeline orchestrator
│   ├── aspect_extract.py    # Explicit extractor (spaCy + LLM)
│   ├── aspect_infer.py      # Implicit weak supervision (RapidFuzz + Semantic)
│   ├── mappings.py          # Universal Superclasses & Symptom Library
│   ├── episodic_builder.py  # ProtoNet episode generator
│   ├── active_learning.py   # Disagreement tracking & HTML Dashboard
│   └── span_model.py        # DeBERTa Evidence Span architecture
├── input/                   # Place raw .csv, .json, or .parquet files here
├── output/                  # Final generated datasets
└── requirements.txt         # Project dependencies
```

---

## 🥑 Getting Started

### 1. Installation
```bash
pip install -r dataset_builder/requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```text
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### 3. Run the Pipeline
Always run from the project root (`ReviewOp`):
```bash
python dataset_builder/code/build_dataset.py --input-dir dataset_builder/input --output-dir dataset_builder/output
```

---

## 📊 Output Schema (Format A: Seq2Seq)
The pipeline produces records with a `target_text` field formatted as:
`aspect | sentiment | evidence ;; aspect | sentiment | evidence`

**Example:**
*"The battery dies by noon and the screen is beautiful."*
👉 `battery_life | negative | battery dies by noon ;; display | positive | screen is beautiful`
