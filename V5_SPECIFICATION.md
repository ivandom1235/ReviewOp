# V5 Research-Grade Specification: Reasoning-Augmented Hybrid Pipeline

## 1. Project Vision

DAGR-PIPE v5 elevates the dataset builder to **Research-Grade** status by introducing a **Reasoning-Augmented Hybrid** architecture. This version solves the "Implicit Aspect Gap" by using Large Language Models (LLMs) as linguistic mediators to bridge the gap between subtle human expressions and grounded extraction rules.

## 2. Core Implementation: Stage B Reasoned Recovery

The v5 pipeline introduces a dual-stage extraction process:

- **Stage A (Heuristic):** Direct keyword matching and stemming (Legacy V4.1 logic).
- **Stage B (Reasoned Recovery):** If Stage A fails, the system invokes an LLM to generate an **Explicit Paraphrase** of the implicit signal. This paraphrase is then re-circulated through the Stage A engine for final grounding.

## 3. Research Novelty & Method Identity

### 3.1 Symbolic-Neural Synthesis

V5 avoids the "black-box" trap by restricting LLMs to a **preprocessing role**. The final decision is always made by the symbolic heuristic engine, preserving the **Grounding-First Inference Contract**.

### 3.2 Domain-Agnostic Reasoning

By using reasoning models to interpret context, the pipeline handles domain-specific nuances (e.g., "The crust was soggy" in food vs. "The screen was dim" in electronics) via a single, domain-neutral prompt-mediator.

## 4. Compute Infrastructure & Fallbacks

To ensure reliability across environments, v5 implements a **Waterfall Fallback Strategy**:

1. **RunPod Serverless (Primary):** High-performance, low-cost H100/A100 inference.
2. **OpenAI/Anthropic (Cloud Fallback):** Reliable API access via standard keys.
3. **Ollama (Local Fallback):** Fully offline processing using local models (e.g., Llama-3-8B).

## 5. Mandatory Ablation Matrix for Research Reporting

To validate the V5 contribution, every build report must evaluate:

- **Baseline:** Stage A (Heuristic) only.
- **V5 Full:** Stage A + Stage B (Hybrid).
- **LLM-Direct:** Asking an LLM to extract JSON without the pipeline (Ablation).
- **No-Grounding:** Reasoning allowed to hallucinate (Ablation).

## 6. Target Quality Gates (V5)

- **Fallback Rate:** < 15% (Reduction from V4.1's 23%).
- **Grounding Rate:** 100% (Ensured by the mediated architecture).
- **Unseen Domain Coverage:** > 65%.
