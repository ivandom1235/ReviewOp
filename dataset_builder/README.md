# Dataset Builder

Builds ReviewOp training/eval datasets from raw CSV/JSON sources.

## What It Produces

- `output/explicit/*.jsonl`: explicit feature rows
- `output/implicit/*.jsonl`: implicit feature rows
- `output/implicit_strict/*.jsonl`: strict implicit subset
- `output/reports/build_report.json`: quality + diagnostics report
- `output/compat/protonet/episodic/*.jsonl`: ProtoNet episodic-compatible export
- `output/compat/protonet/reviewlevel/*.jsonl`: ProtoNet review-level-compatible export

## Prerequisites

From repo root:

```powershell
pip install -r dataset_builder\requirements.txt
```

## Quick Start

From repo root:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

## Safe Preview (No File Writes)

Use this to validate settings quickly:

```powershell
python dataset_builder\code\build_dataset.py --run-profile debug --sample-size 100 --chunk-size 25 --chunk-offset 0 --preview
```

`--preview` (same behavior as `--dry-run`) is non-destructive and does not reset existing outputs.

## Common Command Recipes

### Write a small debug run

```powershell
python dataset_builder\code\build_dataset.py --run-profile debug --sample-size 100 --chunk-size 25 --chunk-offset 0
```

### Emit review template for human annotation

```powershell
python dataset_builder\code\build_dataset.py --emit-review-set --review-set-size 300
```

Template path:

```text
dataset_builder/output/reports/review_set_template.jsonl
```

### Run with gold annotations

```powershell
python dataset_builder\code\build_dataset.py --gold-annotations-path dataset_builder\input\gold_annotations.jsonl
```

### Domain holdout evaluation

```powershell
python dataset_builder\code\build_dataset.py --evaluation-protocol loo --domain-holdout hotel
```

### Disable LLM fallback branch

```powershell
python dataset_builder\code\build_dataset.py --no-enable-llm-fallback
```

## Experiment Runner (`run_experiment.py`)

Preview a planned configuration:

```powershell
python dataset_builder\code\run_experiment.py --plan-only --implicit-mode hybrid --no-drop
```

Run the V4 sweep:

```powershell
python dataset_builder\code\run_experiment.py --execute-v4-sweep
```

Run sweep with custom token ranges:

```powershell
python dataset_builder\code\run_experiment.py --execute-v4-sweep --sweep-implicit-min-tokens 6,8 --sweep-min-text-tokens 3,4
```

Run sweep with gold labels and promote defaults if gates pass:

```powershell
python dataset_builder\code\run_experiment.py --execute-v4-sweep --gold-annotations-path dataset_builder\input\gold_annotations.jsonl --gold-min-rows-for-promotion 600 --apply-best-defaults --no-enable-llm-fallback
```

## Important Behavior

- Write runs reset `output/` by default.
- Preview/dry-run mode never deletes or writes output artifacts.
- Compatibility exports for ProtoNet are always written to:
  - `output/compat/protonet/episodic/`
  - `output/compat/protonet/reviewlevel/`
