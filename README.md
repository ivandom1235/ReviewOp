# ReviewOp Monorepo Guide

This repository has four major parts:
- `frontend/`: UI application.
- `backend/`: API + ABSA inference + analytics.
- `ProtoBackend/`: offline prototype-based implicit aspect benchmarking pipeline.
- `datasetCreator/`: general multi-domain hybrid ABSA dataset creation pipeline.

This README is written so a first-time user can run each part and understand all required inputs and outputs.

## 1) Frontend + Backend (Main App)

> Python requirement for backend: Python `3.13+`

### Quick start (recommended)

Run one-time setup:

```powershell
.\run-project.ps1
```

Then start services:

```powershell
# Terminal 1
cd backend
venv\Scripts\activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm run dev
```

Or use auto-launch:

```powershell
.\run-services.ps1
```

### Backend manual setup

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Before running, update DB credentials in `backend/core/config.py`.

### Seeded accounts
- Admin: `admin` / `12345`
- User: `user` / `12345`

### Core graph endpoints
- `GET /graph/review/{review_id}`: single review explanation graph
- `GET /graph/aspects`: corpus co-occurrence graph

---

## 2) ProtoBackend (Offline Implicit Aspect Benchmark)

`ProtoBackend` is an offline experiment pipeline. It does not run as API service.

It trains prototype vectors from implicit aspect evidence and evaluates/sweeps inference thresholds.

### Input options

ProtoBackend can train from either of these:

1. Default backend JSONL (preferred)
- Files expected:
  - `backend/data/implicit/raw/implicit_reviewlevel_train.jsonl`
  - `backend/data/implicit/raw/implicit_reviewlevel_val.jsonl`
  - `backend/data/implicit/raw/implicit_reviewlevel_test.jsonl`
- Expected JSONL label shape per row:
  - `implicit_labels[*].implicit_aspect`
  - `implicit_labels[*].evidence_sentence`

2. CSV input
- Path example: `ProtoBackend/data/sample_implicit_aspects.csv`
- Required columns:
  - `sentence`
  - `aspect`

### Run commands

Use backend venv python:

```powershell
backend\venv\Scripts\python.exe
```

Train:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/train_proto.py --backend-root . --output-dir ProtoBackend/models/implicit_proto
```

Train from CSV instead:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/train_proto.py --train-csv ProtoBackend/data/sample_implicit_aspects.csv --output-dir ProtoBackend/models/implicit_proto
```

Demo inference:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/run_proto_demo.py --prototypes ProtoBackend/models/implicit_proto/prototypes.npz --sentence "The call keeps dropping and disconnects" --top-k 3 --threshold 0.6
```

Evaluate:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/eval_proto.py --split val --prototypes ProtoBackend/models/implicit_proto/prototypes.npz
backend/venv/Scripts/python.exe ProtoBackend/scripts/eval_proto.py --split test --prototypes ProtoBackend/models/implicit_proto/prototypes.npz
```

Sweep:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/sweep_proto_thresholds.py --split val --prototypes ProtoBackend/models/implicit_proto/prototypes.npz --thresholds 0.45,0.5,0.55,0.6,0.65,0.7 --topks 1,2,3,4,5
```

One-command pipeline:

```powershell
backend/venv/Scripts/python.exe ProtoBackend/scripts/run_proto_pipeline.py --backend-root . --output-dir ProtoBackend/models/implicit_proto --thresholds 0.45,0.5,0.55,0.6,0.65,0.7 --topks 1,2,3,4,5
```

PowerShell shortcut:

```powershell
.\ProtoBackend\run-proto.ps1 pipeline --backend-root . --output-dir ProtoBackend/models/implicit_proto
```

### ProtoBackend outputs (what each file means)

All outputs go to `ProtoBackend/models/implicit_proto/`.

- `prototypes.npz`: learned aspect prototype vectors.
- `label_map.json`: aspect-to-index mapping for prototype matrix.
- `train_data_summary.json`: train dataset class distribution.
- `eval_val.json` / `eval_test.json`: evaluation metrics on selected split.
- `sweep_results_val_<timestamp>.json`: full grid results for `(top_k, threshold)`.
- `best_config.json`: best validation configuration.
- `eval_test_best.json`: test metrics under best validation config.
- `pipeline_report.json`: end-to-end summary of the full pipeline run.

---

## 3) datasetCreator (General Multi-Domain Hybrid ABSA Dataset Pipeline)

`datasetCreator` builds two datasets for the hybrid ABSA method:
- Public normalized dataset: `dataset_public_general.jsonl`
- Augmented synthetic dataset: `dataset_augmented_general.jsonl`

Hybrid flow target:
- Stage A: explicit extraction
- Stage B: implicit symptom inference
- Stage C: evidence-grounded verifier pass

### Input folders

Put files here:
- Public source files: `datasetCreator/data/raw/public/`
- Optional synthetic raw seeds: `datasetCreator/data/raw/synthetic/`

### Public input file types

Accepted formats in `datasetCreator/data/raw/public/`:
- `.csv`
- `.json`
- `.jsonl`

### Public input columns (auto-detected)

For review text (at least one):
- `review_text` or `text` or `review` or `content` or `comment` or `body`

For id (optional):
- `review_id` or `id` or `uid`

For rating (optional):
- `rating` or `stars` or `score`

For domain hint (optional):
- `domain` or `category` or `source` or `dataset` or `vertical`

If `review_id` is missing, one is generated automatically.
If domain is missing/unclear, fallback domain is `shopping`.
Rows with very short review text are dropped.

### Public datasets used

Currently bundled sample:
- `datasetCreator/data/raw/public/sample_public_reviews.jsonl`

Recommended public corpora you can add:
- Amazon Product Reviews
- Yelp Open Dataset
- TripAdvisor Hotel Reviews
- Booking.com Hotel Reviews
- Google Play / App Store review exports
- Kaggle service-review datasets (ride, healthcare, banking, etc.)

### Run dataset creation

From repository root:

```powershell
python datasetCreator/scripts/build_public_dataset.py
python datasetCreator/scripts/build_augmented_dataset.py
python datasetCreator/scripts/merge_and_split.py
python datasetCreator/scripts/run_qa_reports.py
```

One-liner:

```powershell
python datasetCreator/scripts/build_public_dataset.py; python datasetCreator/scripts/build_augmented_dataset.py; python datasetCreator/scripts/merge_and_split.py; python datasetCreator/scripts/run_qa_reports.py
```

Run tests:

```powershell
python -m unittest discover -s datasetCreator/tests -p "test_*.py"
```

### datasetCreator outputs (what each file means)

Root output folders:
- `datasetCreator/data/interim/`
- `datasetCreator/data/processed/`
- `datasetCreator/data/reports/`

Interim:
- `normalized_public.jsonl`: normalized records before strict final filtering.
- `generated_candidates.jsonl`: synthetic generation candidates before final filter.
- `verifier_checked.jsonl`: public records after local verifier checks.

Processed:
- `dataset_public_general.jsonl`: final public dataset.
- `dataset_augmented_general.jsonl`: final augmented dataset.
- `dataset_hybrid_merged.jsonl`: merged public+augmented corpus.
- `train.jsonl`, `dev.jsonl`, `test.jsonl`: split files.
- `test_implicit_only.jsonl`: test subset with implicit labels.
- `test_public_only.jsonl`: test subset from public source only.
- `test_augmented_only.jsonl`: test subset from synthetic source only.
- `test_multiaspect_only.jsonl`: test subset with >1 aspect annotation.
- `test_crossdomain_holdout.jsonl`: held-out domain subset for transfer checks.

Ontology + prompt assets:
- `datasetCreator/data/ontology/canonical_aspects.json`: canonical aspect list + aliases + domain mappings.
- `datasetCreator/data/ontology/symptom_patterns.json`: symptom-to-aspect clues for implicit weak labeling.
- `datasetCreator/prompts/generate_review.txt`: structured synthetic generation prompt contract.
- `datasetCreator/prompts/infer_implicit.txt`: structured implicit inference prompt contract.
- `datasetCreator/prompts/verify_annotation.txt`: structured annotation verification prompt contract.

Reports:
- `domain_stats.json`: number of records per domain.
- `aspect_stats.json`: aspect frequency and sentiment/source stats.
- `qa_report.json`: dedup stats, verifier acceptance/rejection, quality summary.

### Unified output record format

Each processed row is JSON with:
- `review_id`
- `domain`
- `source_type` (`public` or `synthetic`)
- `review_text`
- `annotations[]` with:
  - `aspect`
  - `source` (`explicit` or `implicit`)
  - `sentiment` (`positive|neutral|negative`)
  - `evidence_text`
  - `start_char`, `end_char`
  - `reason_type`
  - `confidence`
- `metadata`

---

## 4) Notes and Guardrails

- Generated/raw datasets are ignored by Git (see `.gitignore`), so data files do not get committed by default.
- `datasetCreator` is dataset-building only; it does not expose API routes.
- `ProtoBackend` is benchmark/offline experimentation only.
- Main app runtime remains `frontend + backend`.
