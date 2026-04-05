# ReviewOp (V5.5 Research-Grade)

ReviewOp is a monorepo for aspect-based sentiment analysis with a **Reasoning-Augmented Hybrid** architecture:

- `backend/`: FastAPI + MySQL APIs for inference, analytics, graph, and user flows
- `frontend/`: React + Vite UI (Performance-optimized, native fetch)
- `dataset_builder/`: High-throughput **Symbolic-Neural Synthesis** pipeline (async)
- `protonet/`: Few-shot Prototypical training/eval/export pipeline

## Technical Vision (V5.5)

ReviewOp V5.5 bridges the "Implicit Aspect Gap" by combining the reliability of symbolic matching with the reasoning power of neural models.

1.  **Stage A (Symbolic):** Heuristic keyword grounding to ensure zero-hallucination.
2.  **Stage B (Neural):** LLM-mediated reasoned recovery for ambiguous or implicit clausal signals.

For more technical depth, see the [Research Overview](SEARCH_OVERVIEW.md).

## Repo Layout

```text
ReviewOp/
|-- backend/
|-- frontend/
|-- dataset_builder/
|-- protonet/
|-- run-project.ps1
|-- run-services.ps1
`-- RESEARCH_OVERVIEW.md
```

## Prerequisites

- Python `3.10` to `3.13`
- Node.js `18+`
- MySQL running locally (or reachable from this machine)

## Quick Start (Windows PowerShell)

### 1. One-time setup

```powershell
Copy-Item .env.example .env
.\run-project.ps1
```

`run-project.ps1` installs backend and frontend dependencies and prepares `backend/venv`.

### 2. Start backend + frontend

```powershell
.\run-services.ps1
```

### 3. Verify

- Backend health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`
- Frontend: URL printed by Vite (usually `http://127.0.0.1:5173`)

## Performance & Scaling

ReviewOp is designed for large-scale research trials:

- **Async Pipeline:** Truly parallel LLM calls in the dataset builder.
- **Vectorized Centroids:** Faster prototypical clustering.
- **AMP Support:** Automatic Mixed Precision for NVIDIA H100/A100.

## Security & Ethics

- **No Axios:** Standardized on `httpx` (Python) and `fetch` (JS) for security.
- **Grounding-First:** Neural models are restricted to preprocessing; decision logic is symbolic and observable.

## Module Docs

- `dataset_builder/README.md`
- `protonet/README.md`
