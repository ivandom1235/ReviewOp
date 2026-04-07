# Dataset Builder (V6)

Builds benchmark files used by `protonet` training and evaluation.

## Outputs (V6)

- `dataset_builder/output/benchmark/ambiguity_grounded/train.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/val.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/test.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/metadata.json`
- `dataset_builder/output/reports/build_report.json`

## Build Dataset

Run from repo root:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

## Preview Only (No Writes)

```powershell
python dataset_builder\code\build_dataset.py --run-profile debug --sample-size 100 --chunk-size 25 --chunk-offset 0 --preview
```

## See Generated Files

```powershell
Get-ChildItem dataset_builder\output\benchmark\ambiguity_grounded
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\train.jsonl -TotalCount 3
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\val.jsonl -TotalCount 3
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\test.jsonl -TotalCount 3
```

## Next Step

Train the model using:

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded
```

## Manual Zip Command (Artifacts Only)

Zip `train.jsonl`, `val.jsonl`, `test.jsonl` plus key reports into `dataset_builder\output\zip`:

```powershell
python dataset_builder\code\build_dataset.py --zip-only --output-dir dataset_builder\output
```
