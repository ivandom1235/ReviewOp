# ReviewOps Dataset Builder

Builds training-ready ABSA datasets (review-level + episodic) with explicit/implicit support.

## Folder Layout
- `dataset_builder/input/`
- `dataset_builder/code/`
- `dataset_builder/output/reviewlevel/{train,val,test}.jsonl`
- `dataset_builder/output/episodic/{train,val,test}.jsonl`
- `dataset_builder/output/reports/{build_report.json, skipped_rows.jsonl}`

## How To Run
From repo root:

```powershell
python dataset_builder/code/build_dataset.py --input-dir dataset_builder/input --output-dir dataset_builder/output
```

Compatibility entrypoint (accepts `--input`, `--output`, `--use-openai`):

```powershell
python dataset_builder/code/main.py --input ./dataset_builder/input --output ./dataset_builder/output --use-openai false
```

You can also point to any other input folder:

```powershell
python dataset_builder/code/build_dataset.py --input-dir . --output-dir dataset_builder/output
```

## Dry Run (No Files Written)
```powershell
python dataset_builder/code/build_dataset.py --input-dir dataset_builder/input --dry-run
```

## Clean Outputs
Remove all generated output artifacts:

```powershell
python dataset_builder/code/clean_outputs.py --output-dir dataset_builder/output
```

This command removes the output folder and recreates a clean empty structure.

If you want removal without recreating subfolders:

```powershell
python dataset_builder/code/clean_outputs.py --output-dir dataset_builder/output --no-recreate
```

## Important Behavior
- If `output/` or any subfolder is missing, `build_dataset.py` recreates:
  - `output/reviewlevel`
  - `output/episodic`
  - `output/reports`
- Split handling:
  - uses existing split if present (`train/val/test`)
  - otherwise creates grouped split by review id (default `0.8/0.1/0.1`)
- Episodic rows include both `aspect` and `implicit_aspect` for compatibility.
- Evidence extraction prefers `evidence` column, then `from/to` span text when available, then sentence-level fallback.
- Sentiment normalization maps missing/unknown-like values to `neutral` (no `unknown` labels in output).

## Common Flags
- `--split-ratios 0.8,0.1,0.1`
- `--max-aspects 5`
- `--confidence-threshold 0.35`
- `--prefer-open-aspect`
- `--seed 42`

## Quick Verification
After a run, check these files exist:
- `dataset_builder/output/reviewlevel/train.jsonl`
- `dataset_builder/output/reviewlevel/val.jsonl`
- `dataset_builder/output/reviewlevel/test.jsonl`
- `dataset_builder/output/episodic/train.jsonl`
- `dataset_builder/output/episodic/val.jsonl`
- `dataset_builder/output/episodic/test.jsonl`
- `dataset_builder/output/reports/build_report.json`
