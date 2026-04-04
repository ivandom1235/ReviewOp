# ReviewOp

ReviewOp is a monorepo for aspect-based sentiment analysis with:

- `backend/`: FastAPI + MySQL APIs for inference, analytics, graph, and user flows
- `frontend/`: React + Vite UI
- `dataset_builder/`: dataset generation pipeline (explicit + implicit JSONL outputs)
- `protonet/`: few-shot ProtoNet training/eval/export pipeline

## Repo Layout

```text
ReviewOp/
|-- backend/
|-- frontend/
|-- dataset_builder/
|-- protonet/
|-- run-project.ps1
`-- run-services.ps1
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

Or run manually in two terminals:

```powershell
# Terminal 1
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2
cd frontend
npm run dev
```

### 3. Verify

- Backend health: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`
- Frontend: URL printed by Vite (usually `http://127.0.0.1:5173`)

## Environment Setup

Edit `.env` (repo root) and set at least:

```dotenv
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=protodb
```

Optional LLM/provider settings are in `.env.example`.

## Common Workflows

### Build datasets

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

Key outputs:

- `dataset_builder/output/explicit/*.jsonl`
- `dataset_builder/output/implicit/*.jsonl`
- `dataset_builder/output/implicit_strict/*.jsonl`
- `dataset_builder/output/reports/build_report.json`
- `dataset_builder/output/compat/protonet/episodic/*.jsonl`
- `dataset_builder/output/compat/protonet/reviewlevel/*.jsonl`

### Train ProtoNet from dataset builder output

```powershell
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

If model weights are not cached yet:

```powershell
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer --allow-model-download
```

## Module Docs

- `dataset_builder/README.md`
- `protonet/README.md`
