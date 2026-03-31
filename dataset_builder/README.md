# Dataset Builder

Clean-room English-only dataset builder for explicit and implicit review features.

## Layout

- `dataset_builder/input/` raw input files
- `dataset_builder/output/explicit/` explicit feature exports
- `dataset_builder/output/implicit/` implicit feature exports
- `dataset_builder/output/reports/` build reports and diagnostics

## Run

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

Chunked prototyping:

```powershell
python dataset_builder\code\build_dataset.py --sample-size 100 --chunk-size 25 --chunk-offset 0 --preview
```
