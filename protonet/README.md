# Protonet

`protonet/` is the standalone few-shot training pipeline for ReviewOp. It can train from:

- episodic example rows in `input/episodic/`
- review-level rows in `input/reviewlevel/`

The entry point is `code/cli.py`, which supports `train`, `eval`, and `export`.

## Directory Layout

```text
protonet/
|-- code/
|-- input/
|   |-- episodic/
|   `-- reviewlevel/
|-- metadata/
|-- output/
`-- requirements.txt
```

## Installation

From the repo root:

```powershell
pip install -r protonet\requirements.txt
```

Or from inside `protonet/`:

```powershell
pip install -r requirements.txt
```

## Input Modes

### Episodic input

This is the primary path for the repo. Copy the dataset builder output into `protonet/input/episodic/`:

```powershell
Copy-Item "dataset_builder\output\episodic\*" -Destination "protonet\input\episodic\" -Recurse -Force
```

If you are already inside `protonet/`, use:

```powershell
Copy-Item "..\dataset_builder\output\episodic\*" -Destination ".\input\episodic\" -Recurse -Force
```

Expected files:

- `protonet/input/episodic/train.jsonl`
- `protonet/input/episodic/val.jsonl`
- `protonet/input/episodic/test.jsonl`

### Review-level input

You can also train from review-level rows:

```powershell
Copy-Item "dataset_builder\output\reviewlevel\*" -Destination "protonet\input\reviewlevel\" -Recurse -Force
```

If you are already inside `protonet/`, use:

```powershell
Copy-Item "..\dataset_builder\output\reviewlevel\*" -Destination ".\input\reviewlevel\" -Recurse -Force
```

The CLI will adapt review-level rows into the format needed for episode generation.

## CLI Commands

### Train

#### From the repo root

Default transformer-oriented training run:

```powershell
python protonet\code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

Lower-cost CPU-friendly run:

```powershell
python protonet\code\cli.py train --input-type episodic --encoder-backend bow --n-way 2 --k-shot 1 --q-query 1 --epochs 2 --patience 1 --max-train-episodes 40 --max-eval-episodes 16
```

Transformer run with model download enabled:

```powershell
python protonet\code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --encoder-model-name distilroberta-base --production-require-transformer --allow-model-download
```

Train from review-level input:

```powershell
python protonet\code\cli.py train --input-type reviewlevel --encoder-backend transformer --production-require-transformer
```

#### From inside `protonet/`

Default episodic run:

```powershell
cd protonet
python code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

Review-level run:

```powershell
cd protonet
python code\cli.py train --input-type reviewlevel --encoder-backend transformer --production-require-transformer
```

### Evaluate

Evaluate a saved checkpoint on a split:

```powershell
python protonet\code\cli.py eval --input-type episodic --split test --checkpoint protonet\output\checkpoints\best.pt
```

Inside `protonet/`:

```powershell
python code\cli.py eval --input-type episodic --split test --checkpoint output\checkpoints\best.pt
```

### Export

Export a portable model bundle from a checkpoint:

```powershell
python protonet\code\cli.py export --input-type episodic --checkpoint protonet\output\checkpoints\best.pt
```

Inside `protonet/`:

```powershell
python code\cli.py export --input-type episodic --checkpoint output\checkpoints\best.pt
```

## Important Options

Common flags exposed by `code/cli.py`:

```text
--input-type {episodic,reviewlevel}
--input-dir
--output-dir
--metadata-dir
--encoder-backend {auto,transformer,bow}
--encoder-model-name
--n-way
--k-shot
--q-query
--epochs
--warmup-epochs
--patience
--max-train-episodes
--max-eval-episodes
--contrastive-weight
--prototype-smoothing
--low-confidence-threshold
--seed
--no-progress
--force-rebuild-episodes
--strict-encoder
--production-require-transformer
--allow-model-download
```

Current default transformer model:

```text
microsoft/deberta-v3-base
```

## Outputs

Training writes artifacts to both `output/` and `metadata/`.

### `output/`

- `output/checkpoints/best.pt`
- `output/episodes/`
- `output/predictions/val_predictions.jsonl`
- `output/predictions/test_predictions.jsonl`
- `output/training_history.json`

### `metadata/`

- `metadata/report.json`
- `metadata/model_bundle.pt`

`metadata/report.json` summarizes:

- resolved config
- input split sizes
- generated episode counts
- train, validation, and test metrics
- exported artifact paths



## Typical End-to-End Flow

```powershell
python dataset_builder\code\build_dataset.py
Copy-Item "dataset_builder\output\episodic\*" -Destination "protonet\input\episodic\" -Recurse -Force
python protonet\code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```

That flow leaves the trained checkpoint in `protonet/output/checkpoints/` and the portable bundle plus report in `protonet/metadata/`.

## Fastest Correct Commands

If you only want the shortest working sequence for this repo:

From the repo root:

```powershell
python dataset_builder\code\build_dataset.py
Copy-Item "dataset_builder\output\episodic\*" -Destination "protonet\input\episodic\" -Recurse -Force
python protonet\code\cli.py train --input-type episodic --force-rebuild-episodes --encoder-backend transformer --production-require-transformer
```
