# ReviewOp (V6)

ReviewOp is a monorepo for aspect-based sentiment analysis.

- `dataset_builder/`: builds benchmark datasets
- `protonet/`: trains/evaluates/exports ProtoNet
- `backend/`: FastAPI APIs
- `frontend/`: React + Vite UI

## Repo Layout

```text
ReviewOp/
|-- backend/
|-- frontend/
|-- dataset_builder/
|-- protonet/
|-- run-project.ps1
|-- run-services.ps1
`-- SEARCH_OVERVIEW.md
```

## Quick Start (Root)

Run from repository root in PowerShell.

```powershell
# 1) Prepare backend/frontend dependencies
.\run-project.ps1

# 2) Build dataset
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output

# 3) Train model
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded

# 4) Evaluate best checkpoint
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --split test --checkpoint protonet\output\checkpoints\best.pt

# 5) Export runtime bundle
python protonet\code\cli.py export --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt

# 6) Start backend + frontend (new terminals)
.\run-services.ps1
```

## RunPod Topology

- `dataset_builder` uses its own RunPod LLM endpoint via `REVIEWOP_RUNPOD_API_KEY` and `REVIEWOP_RUNPOD_ENDPOINT_URL`
- `protonet` and `backend` share one RunPod Flash endpoint via `PROTONET_FLASH_ENDPOINT_ID`
- local fallback remains available through the exported Protonet bundle at `REVIEWOP_PROTONET_BUNDLE_PATH`

## Root Commands And Options

### `run-project.ps1`

Purpose: setup helper (checks Python/Node, creates backend venv, installs backend/frontend deps).

Options: none.

### `run-services.ps1`

Purpose: starts backend (`uvicorn`) and frontend (`npm run dev`) in new PowerShell windows.

Options: none.

### `npm test`

Purpose: runs the repo-wide test suite from the root.

Steps:

- backend unittest discovery
- dataset_builder unittest discovery
- protonet unittest discovery
- frontend production build

Individual entries:

- `npm run test:backend`
- `npm run test:dataset-builder`
- `npm run test:protonet`
- `npm run test:frontend`

### `dataset_builder` CLI

Command:

```powershell
python dataset_builder\code\build_dataset.py [options]
```

Full options: `dataset_builder/README.md`

### `protonet` CLI

Commands:

```powershell
python protonet\code\cli.py train [options]
python protonet\code\cli.py eval [common options] [eval options]
python protonet\code\cli.py export [common options] [export options]
```

Full options: `protonet/README.md`
