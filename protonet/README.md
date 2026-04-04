# ProtoNet

Standalone few-shot training pipeline for ReviewOp.

CLI entrypoint: `protonet/code/cli.py`

Supported commands:

- `train`
- `eval`
- `export`

## Install

From repo root:

```powershell
pip install -r protonet\requirements.txt
```

## Recommended Input Source

Use dataset builder compatibility exports directly (no copy needed):

- Episodic: `dataset_builder/output/compat/protonet/episodic/`
- Review-level: `dataset_builder/output/compat/protonet/reviewlevel/`

## Important Path Clarification

- `--input-dir` points to where ProtoNet reads data from (for example `dataset_builder/output/...`).
- ProtoNet output is **not** written into `dataset_builder/output/`.
- By default, ProtoNet writes to:
  - `protonet/output/` (checkpoints, episode cache, predictions, history)
  - `protonet/metadata/` (report and exported bundle)
- You only change this behavior if you explicitly pass `--output-dir` or `--metadata-dir`.

## Train

### Production-style transformer run

```powershell
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

If model download is required:

```powershell
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer --allow-model-download
```

### Fast CPU-friendly smoke run

```powershell
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --encoder-backend bow --n-way 2 --k-shot 1 --q-query 1 --epochs 2 --patience 1 --max-train-episodes 40 --max-eval-episodes 16
```

### Train from review-level export

```powershell
python protonet\code\cli.py train --input-type reviewlevel --input-dir dataset_builder\output\compat\protonet\reviewlevel --encoder-backend transformer --production-require-transformer
```

## Evaluate

```powershell
python protonet\code\cli.py eval --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --split test --checkpoint protonet\output\checkpoints\best.pt
```

## Export

```powershell
python protonet\code\cli.py export --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --checkpoint protonet\output\checkpoints\best.pt
```

## Key CLI Options

```text
--input-type {episodic,reviewlevel}
--input-dir
--output-dir
--metadata-dir
--encoder-backend {auto,transformer,bow}
--encoder-model-name
--force-rebuild-episodes
--production-require-transformer
--allow-model-download
```

Default transformer model: `microsoft/deberta-v3-base`

## Outputs

- `protonet/output/checkpoints/best.pt`
- `protonet/output/predictions/*.jsonl`
- `protonet/output/training_history.json`
- `protonet/metadata/report.json`
- `protonet/metadata/model_bundle.pt`

## End-to-End Commands (Repo Root)

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
python protonet\code\cli.py train --input-type episodic --input-dir dataset_builder\output\compat\protonet\episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```
