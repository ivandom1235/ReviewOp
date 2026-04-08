# ProtoNet (V6)

ProtoNet trains, evaluates, and exports the ReviewOp ambiguity-aware aspect model.

## Commands

Run from repo root:

```powershell
python protonet\code\cli.py train [common options]
python protonet\code\cli.py eval [common options] [eval options]
python protonet\code\cli.py export [common options] [export options]
```

## Quick Examples

```powershell
# Train
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded

# Evaluate (test split by default)
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt

# Export model bundle
python protonet\code\cli.py export --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt
```

## Inference Service

The runtime bundle can be served locally through the FastAPI app or imported directly by the backend.

Local HTTP service:

```powershell
python -m uvicorn protonet.http_api:app --host 127.0.0.1 --port 8011
```

Environment variables:

- `REVIEWOP_PROTONET_BUNDLE_PATH` overrides the bundle location
- `REVIEWOP_PROTONET_REQUEST_TIMEOUT_SECONDS` sets the backend HTTP timeout

Training and evaluation run locally:

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded --checkpoint protonet\output\checkpoints\best.pt
```

## Common Options (all commands)

- `--input-type` choices: `benchmark` (default: `benchmark`)
- `--input-dir` path to benchmark input directory
- `--output-dir` path for checkpoints/predictions/reports
- `--metadata-dir` path for metadata outputs
- `--encoder-backend` choices: `auto`, `transformer`, `bow` (default: `auto`)
- `--encoder-model-name` model id (default: `microsoft/deberta-v3-base`)
- `--n-way` episode N-way (default: `3`)
- `--k-shot` episode K-shot (default: `2`)
- `--q-query` episode query count (default: `2`)
- `--epochs` training epochs (default: `8`)
- `--learning-rate` head learning rate (default: `5e-4`)
- `--encoder-learning-rate` encoder learning rate (default: `1e-5`)
- `--warmup-epochs` warmup epochs (default: `1`)
- `--patience` early-stopping patience (default: `3`)
- `--max-train-episodes` max train episodes (default: `120`)
- `--max-eval-episodes` max eval episodes (default: `48`)
- `--contrastive-weight` contrastive loss weight (default: `0.15`)
- `--prototype-smoothing` prototype smoothing factor (default: `0.05`)
- `--low-confidence-threshold` low-confidence cutoff (default: `0.55`)
- `--selective-alpha` selective weighting alpha (default: `0.6`)
- `--selective-beta` selective weighting beta (default: `0.25`)
- `--selective-gamma` selective weighting gamma (default: `0.1`)
- `--selective-delta` selective weighting delta (default: `0.05`)
- `--abstain-threshold` abstain threshold (default: `0.55`)
- `--multi-label-margin` multi-label margin (default: `0.08`)
- `--sentiment-pipeline` choices: `joint`, `post_aspect`, `both` (default: `both`)
- `--novelty-threshold` global novelty threshold (default: `0.45`)
- `--novelty-known-threshold` known threshold (default: `0.35`)
- `--novelty-novel-threshold` novel threshold (default: `0.65`)
- `--novelty-calibration-path` path to calibration JSON
- `--seed` random seed (default: `42`)
- `--no-progress` disable progress display
- `--force-rebuild-episodes` ignore cache and rebuild episode sets
- `--strict-encoder` fail if preferred encoder backend is unavailable
- `--production-require-transformer` enforce transformer backend for production paths
- `--allow-model-download` allow downloading model weights if missing
- `--compile-model` enable model compilation path

## Eval Options

- `--split` choices: `train`, `val`, `test` (default: `test`)
- `--checkpoint` checkpoint path (default: `<output-dir>\checkpoints\best.pt`)

## Export Options

- `--checkpoint` checkpoint path (default: `<output-dir>\checkpoints\best.pt`)
