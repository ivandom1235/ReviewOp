# ReviewOp

ReviewOp is a research and demo system for aspect-based sentiment analysis on product reviews. The project combines a dataset-building pipeline, a ProtoNet implicit-aspect classifier, a FastAPI backend, and a React frontend for admin and user-facing review workflows.

## System Flow

```text
raw / scraped reviews
    -> dataset_builder
    -> benchmark JSONL artifacts
    -> protonet training and model export
    -> backend hybrid inference, analytics, and knowledge graph APIs
    -> frontend admin and user portals
```

The main runtime model path is:

1. `dataset_builder` creates canonical train/validation/test JSONL files.
2. `protonet` trains and exports `protonet/metadata/model_bundle.pt`.
3. `backend` loads the exported bundle when implicit-aspect inference is enabled.
4. `frontend` calls the backend APIs for authentication, review submission, analytics, graph exploration, jobs, and exports.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `backend/` | FastAPI app, route modules, database access, auth dependencies, analytics services, batch jobs, inference orchestration, and export APIs. |
| `frontend/` | Vite React app for admin dashboards, product/review pages, authentication, user review submission, and analytics visualizations. |
| `dataset_builder/` | Async pipeline for turning raw reviews into benchmark-ready aspect/sentiment examples, including optional LLM-backed ambiguity resolution. |
| `protonet/` | ProtoNet training, evaluation, export, and runtime serving code for implicit aspect classification. |
| `metadata/` | Shared metadata artifacts used by the app and model pipelines. |
| `plans/`, `plan.md`, `tasks.md` | Local implementation planning and task tracking artifacts. |
| `run-project.ps1` | Windows setup script for backend and frontend dependencies. |
| `run-services.ps1` | Windows helper script that starts backend and frontend dev servers. |

Additional architecture notes may exist outside the repository under `../agent_docs/ReviewOp/`.

## Quick Start

Prerequisites:

- Python 3.10 through 3.13.
- Node.js and npm.
- MySQL-compatible database configured for the backend.
- PowerShell on Windows for the included helper scripts.

Install dependencies:

```powershell
.\run-project.ps1
```

Start the backend and frontend:

```powershell
.\run-services.ps1
```

Manual service startup:

```powershell
.\backend\venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

```powershell
npm --prefix frontend run dev
```

Default local URLs:

- Backend API docs: `http://127.0.0.1:8000/docs`
- Frontend dev server: printed by Vite, usually `http://127.0.0.1:5173`

## Core Workflows

Run the commands in this section from the repository root.

### Build Dataset Artifacts

```powershell
python dataset_builder\scripts\build_benchmark.py dataset_builder\input --output-dir dataset_builder\output --sample-size 50 --llm none --overwrite
```

Expected benchmark outputs:

- `dataset_builder/output/benchmark/ambiguity_grounded/train.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/val.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/test.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/metadata.json`

For LLM-backed experiments, prefer the experiment runner documented in `dataset_builder/README.md` so caches, reports, and quality gates remain reproducible.

### Train and Export ProtoNet

```powershell
python -m protonet.cli compare --artifact-dir dataset_builder\output --output-dir protonet\output --split test
```

The backend expects the exported bundle at `protonet/metadata/model_bundle.pt` unless configured otherwise.

### Run Frontend Build

```powershell
npm --prefix frontend run build
```

### Run Backend Import Check

```powershell
.\backend\venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'backend'); import app; print('backend import ok')"
```

## Runtime Configuration

The backend uses environment variables and configuration in `backend/core/config.py`. Do not hardcode credentials in code or documentation.

Important runtime switches:

| Variable | Purpose |
| --- | --- |
| `REVIEWOP_ENABLE_IMPLICIT` | Enables or disables ProtoNet implicit-aspect inference in the backend. |
| `REVIEWOP_TRUSTED_BUNDLE_ROOTS` | Adds trusted directories for loading ProtoNet bundles. |

Admin APIs are protected by backend authentication dependencies. User portal routes use user-token authentication. Local demo/default accounts should only be enabled in dev/demo/local environments.

## Security Notes

- Never commit `.env`, database credentials, tokens, or model-provider API keys.
- Backend export endpoints require admin authentication.
- ProtoNet bundle loading is restricted to trusted roots because exported PyTorch bundles require object deserialization.
- Batch CSV processing runs asynchronously to avoid blocking API requests.

## Folder Documentation

- `backend/README.md`
- `frontend/README.md`
- `dataset_builder/README.md`
- `protonet/`

Read the folder-level README before changing that subsystem.
