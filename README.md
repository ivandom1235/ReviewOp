# ReviewOp (V6)

ReviewOp is a monorepo for aspect-based sentiment analysis.

- `dataset_builder/`: builds V6 benchmark datasets
- `protonet/`: trains/evaluates/exports the V6 ProtoNet model
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

## V6 End-to-End (Train + See Outputs)

Run from repository root in PowerShell.

### 1) Build V6 dataset artifacts

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

### 2) Train ProtoNet on V6 benchmark

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld
```

### 3) Evaluate best checkpoint

```powershell
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld --split test --checkpoint protonet\output\checkpoints\best.pt
```

### 4) Export runtime bundle

```powershell
python protonet\code\cli.py export --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld --checkpoint protonet\output\checkpoints\best.pt
```

### 5) See generated files

```powershell
Get-ChildItem dataset_builder\output\benchmark\ambiguity_openworld
Get-ChildItem protonet\output -Recurse
```

### 6) Manually zip dataset artifacts (train/val/test + reports)

```powershell
python dataset_builder\code\build_dataset.py --zip-only --output-dir dataset_builder\output
```

This creates a zip file in `dataset_builder\output\zip\`.

Module-level details:

- `dataset_builder/README.md`
- `protonet/README.md`
