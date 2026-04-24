# ProtoNet

`protonet` is the core Few-Shot Learning (FSL) engine for ReviewOp. It trains and serves Prototypical Networks for implicit aspect classification, featuring selective routing for novelty detection and joint aspect-sentiment modeling.

## Architecture & Code Map

The system follows a clean, CLI-driven pipeline from benchmark ingestion to production export:

| Component | Responsibility |
| --- | --- |
| `code/cli.py` | Unified entry point for `train`, `eval`, and `export` commands. |
| `code/trainer.py` | Training loop with episodic sampling, contrastive learning, and early stopping. |
| `code/evaluator.py` | Multi-split evaluation with selective routing (abstain/novel/known) logic. |
| `code/model.py` | ProtoNet architecture with flexible encoder backends and projection heads. |
| `code/encoder.py` | Sentence embeddings via Transformers (DeBERTa-v3) or fast Bag-of-Words (BoW). |
| `code/dataset_reader.py` | Robust ingestion of ReviewOp Benchmark artifacts (`manifest.json` aware). |
| `code/episode_builder.py` | Deterministic N-way K-shot episode construction with joint-label support. |
| `code/selective_decisions.py` | Logic for selective routing based on distance, energy, and ambiguity. |
| `code/export_bundle.py` | Packaging of weights, prototype banks, and calibration for production. |
| `code/calibrate_novelty.py` | Automatic threshold optimization for "None of the Above" (novelty) detection. |
| `code/runtime_infer.py` | High-performance inference engine for backend integration. |

## Key Features

- **Joint Labeling**: Predicts `aspect__sentiment` pairs as single semantic units.
- **Selective Routing**: Implements a three-band decision model:
    - **Known**: Confident prediction within the supported label set.
    - **Novel**: Confident "None of the Above" detection (Energy/Distance based).
    - **Abstain**: Uncertain or ambiguous examples (Ambiguity/Confidence based).
- **Hardness Awareness**: Pipeline is aware of H0-H3 hardness tiers in the benchmark.
- **Multi-Split Evaluation**: Automatically reports metrics for Random, Grouped, and Domain Holdout protocols.

## Inputs & Outputs

### Inputs
Consumes standard artifacts from `dataset_builder/output/`:
- `train.jsonl`, `val.jsonl`, `test.jsonl`
- `manifest.json` (Required for contract verification)

### Outputs
- `protonet/output/checkpoints/`: Model weights (`best.pt`).
- `protonet/output/predictions/`: Detailed inference logs including novelty scores.
- `protonet/metadata/model_bundle.pt`: Unified artifact for production deployment.
- `protonet/metadata/novelty_calibration_v2.json`: Optimized routing thresholds.

## Workflow Commands

### 1. Train
Trains the model and automatically calibrates novelty thresholds:
```powershell
python protonet\code\cli.py train --input-dir dataset_builder\output --epochs 12 --n-way 3 --k-shot 2
```

### 2. Evaluate
Run detailed evaluation on a specific split:
```powershell
python protonet\code\cli.py eval --input-dir dataset_builder\output --checkpoint protonet\output\checkpoints\best.pt --split test
```

### 3. Export
Manually re-export a production bundle:
```powershell
python protonet\code\cli.py export --checkpoint protonet\output\checkpoints\best.pt
```

## Advanced Configuration

The CLI supports extensive hyperparameter tuning:
- `--encoder-backend`: Choose between `auto`, `transformer`, or `bow`.
- `--contrastive-weight`: Balance between classification and embedding separation.
- `--novelty-threshold`: Base threshold for novelty detection.
- `--low-confidence-threshold`: Threshold for the "Abstain" band.
- `--multi-label-margin`: Margin for detecting multi-interpretation ambiguity.

## Development

### Unit Tests
```powershell
pytest protonet/tests
```

### Import Verification
```powershell
python -c "from protonet.code.runtime_infer import ProtonetRuntime; print('Protonet Runtime OK')"
```
