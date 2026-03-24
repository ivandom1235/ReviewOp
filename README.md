# ReviewOp

ReviewOp is a monorepo for aspect-based sentiment analysis workflows across four main areas:

- `frontend/`: React + Vite UI for admin analytics and the user review portal.
- `backend/`: FastAPI + MySQL application for inference, analytics, graph views, jobs, and user flows.
- `dataset_builder/`: Offline pipeline that converts raw review datasets into normalized review-level and episodic JSONL outputs.
- `protonet/`: Standalone prototypical network training and export pipeline for few-shot experiments.

The current repo layout uses the active `protonet/` module for the implicit prototype pipeline.

## Repo Layout

```text
ReviewOp/
|-- backend/
|-- dataset_builder/
|-- frontend/
|-- protonet/
|-- run-project.ps1
`-- run-services.ps1
```

## Quick Start

### Automated setup

From the repo root:

```powershell
.\run-project.ps1
```

This script:

- checks for Python and Node.js
- creates `backend/venv` if needed
- installs backend Python dependencies
- installs frontend npm dependencies
- pauses so you can confirm backend database settings

After setup, start both services:

```powershell
.\run-services.ps1
```

### Manual setup

Backend:

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

## Configuration

### Backend

The backend reads environment variables from `backend/.env`. Start from `backend/.env.example` and update the MySQL values:

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_local_password_here
MYSQL_DB=protodb
```

When those credentials are valid, the backend will create the configured database automatically if it does not already exist.

Useful backend defaults:

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Default seq2seq model: `google/flan-t5-small`

Seeded accounts created on startup:

- Admin: `admin` / `12345`
- User: `user` / `12345`

### Frontend

The Vite dev server proxies API requests to `http://127.0.0.1:8000` by default. You can override that with:

- `VITE_PROXY_TARGET`
- `VITE_API_BASE_URL`

## Data and Model Workflow

### 1. Build datasets

Place raw files in `dataset_builder/input/`, then run:

```powershell
cd dataset_builder\code
python build_dataset.py
```

This writes:

- review-level data to `dataset_builder/output/reviewlevel/`
- episodic data to `dataset_builder/output/episodic/`
- quality reports to `dataset_builder/output/reports/`

See `dataset_builder/README.md` for CLI flags and output details.

### 2. Train the ProtoNet pipeline

The ProtoNet module can train from either episodic or review-level JSONL input. For the common episodic flow:

```powershell
Copy-Item "dataset_builder\output\episodic\*" -Destination "protonet\input\episodic\" -Recurse -Force
python protonet\code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

Training artifacts are written under:

- `protonet/output/`
- `protonet/metadata/`

See `protonet/README.md` for `train`, `eval`, and `export` commands.

## Current Application Surface

Backend routes currently include:

- `/infer`
- `/jobs`
- `/analytics`
- `/graph`
- `/user`

The frontend includes:

- admin dashboards and analytics views
- graph exploration screens
- user authentication and review submission flows
- product search and review history pages

## Module Docs

- `dataset_builder/README.md`
- `protonet/README.md`

## Notes

- Generated datasets, checkpoints, and local artifacts should stay out of version control unless you intentionally want to track them.
- Some training and inference flows may download model weights the first time they run.
- Temporary folders may be created by tests under `protonet/`; they are not part of the stable input/output contract.
