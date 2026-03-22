# ProtoBackend

Offline prototype-based implicit aspect detector with one CLI and one input contract.

## Folder layout
- `implicit_proto/` core model and evaluation modules.
- `input/reviewlevel/{train,val,test}.jsonl` review-level dataset files.
- `input/episodic/{train,val,test}.jsonl` episodic dataset files.
- `outputs/<dataset_family>/` generated artifacts and reports.
- `proto_cli.py` single entrypoint for train/eval/sweep/run/predict.

## Requirements
Use backend venv Python:

```bash
backend/venv/Scripts/python.exe
```

## Input contract
Place files exactly as:

```text
ProtoBackend/input/
  reviewlevel/
    train.jsonl
    val.jsonl
    test.jsonl
  episodic/
    train.jsonl
    val.jsonl
    test.jsonl
```

Each JSONL row is backend-compatible (contains `implicit_labels[*].implicit_aspect` and `implicit_labels[*].evidence_sentence`).

## Quick start

### Full pipeline (default): train + val sweep + best test eval
Reviewlevel:
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py run --dataset-family reviewlevel
```

Episodic:
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py run --dataset-family episodic
```

You can omit `run` because it is the default command:
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py --dataset-family reviewlevel
```

### Train only
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py train --dataset-family reviewlevel
```

### Evaluate val/test split
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py eval --dataset-family reviewlevel --split val
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py eval --dataset-family reviewlevel --split test
```

### Sweep threshold/top-k
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py sweep --dataset-family reviewlevel --split val --thresholds 0.45,0.5,0.55,0.6,0.65,0.7 --topks 1,2,3,4,5
```

### Predict one sentence
```bash
backend/venv/Scripts/python.exe ProtoBackend/proto_cli.py predict --dataset-family reviewlevel --sentence "The call keeps showing busy and disconnects" --top-k 3 --threshold 0.6
```

## Outputs
Per family output directory:

```text
ProtoBackend/outputs/<dataset_family>/
  prototypes.npz
  label_map.json
  encoder_model/
  train_summary.json
  train_data_summary.json
  sweep_results_val_<timestamp>.json
  best_config.json
  eval_test_best.json
  pipeline_report.json
  artifact_manifest.json
  backend_artifacts.json
```

- `artifact_manifest.json`: existence + file size metadata for all core output artifacts.
- `backend_artifacts.json`: backend-friendly pointers to key artifacts (`prototypes`, `label_map`, `encoder_model`, `best_config`, and `eval_test_best`).

## Migration from old scripts
- `scripts/train_proto.py` -> `proto_cli.py train`
- `scripts/eval_proto.py` -> `proto_cli.py eval`
- `scripts/sweep_proto_thresholds.py` -> `proto_cli.py sweep`
- `scripts/run_proto_pipeline.py` -> `proto_cli.py run`
- `scripts/run_proto_demo.py` -> `proto_cli.py predict`

## Cleanup Artifacts
Use the cleanup utility to remove generated outputs and prune related input JSONL files to core splits only.

Default cleanup (removes `ProtoBackend/outputs` and prunes JSONLs under `ProtoBackend/input`):
```bash
backend/venv/Scripts/python.exe ProtoBackend/clean_outputs.py
```

Cleanup without recreating empty output folders:
```bash
backend/venv/Scripts/python.exe ProtoBackend/clean_outputs.py --no-recreate
```

Skip JSONL pruning:
```bash
backend/venv/Scripts/python.exe ProtoBackend/clean_outputs.py --no-prune-jsonl
```

Core JSONL files kept by default:
- `ProtoBackend/input/episodic/train.jsonl`
- `ProtoBackend/input/episodic/val.jsonl`
- `ProtoBackend/input/episodic/test.jsonl`
- `ProtoBackend/input/reviewlevel/train.jsonl`
- `ProtoBackend/input/reviewlevel/val.jsonl`
- `ProtoBackend/input/reviewlevel/test.jsonl`
