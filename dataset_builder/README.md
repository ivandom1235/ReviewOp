# ReviewOp Dataset Builder

The dataset builder turns raw review files into two downstream-friendly JSONL formats:

- `reviewlevel/`: one row per review, with normalized labels plus a `target_text` string
- `episodic/`: one row per labeled example, ready for few-shot episode generation and ProtoNet training

It is the bridge between messy raw source files and the normalized artifacts used by the rest of the repo.

## What It Does

The current pipeline in `code/build_dataset.py` handles:

- schema detection for supported raw review files
- domain inference when a source file does not provide one
- hybrid explicit and implicit aspect labeling
- evidence span extraction for each label
- open-aspect compaction and canonical label normalization
- class balancing for episodic outputs
- diagnostics and readiness reports under `output/reports/`

The builder also produces a balanced episodic training split at `output/episodic/train_balanced.jsonl`.

## Supported Input Files

The builder scans the input directory for these file types:

- `.csv`
- `.tsv`
- `.json`
- `.jsonl`
- `.xlsx`
- `.xls`
- `.xml`
- `.gz`

By default, the CLI reads from `dataset_builder/input/`.

## Project Layout

```text
dataset_builder/
|-- code/
|   |-- build_dataset.py
|   |-- aspect_infer.py
|   |-- aspect_extract.py
|   |-- evidence_extract.py
|   |-- episodic_builder.py
|   |-- quality_diagnostics.py
|   `-- senticnet_utils.py
|-- input/
|-- output/
|-- resources/
`-- requirements.txt
```

## Installation

From the repo root:

```powershell
pip install -r dataset_builder\requirements.txt
```

Or from inside `dataset_builder/`:

```powershell
pip install -r requirements.txt
```

If your environment does not already have the spaCy English model used by the extraction stack, install it separately:

```powershell
python -m spacy download en_core_web_sm
```

## Running the Builder

### From the repo root

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

### From inside `dataset_builder/`

```powershell
cd dataset_builder
python code\build_dataset.py --input-dir input --output-dir output
```

### Minimal default run

If your files are already under `dataset_builder/input/`, this is enough:

```powershell
cd dataset_builder
python code\build_dataset.py
```

### Common flags

```text
--input-dir
--output-dir
--dry-run
--split-ratios
--max-aspects
--confidence-threshold
--prefer-open-aspect
--domain-agnostic-mode {auto,off,always}
--senticnet / --no-senticnet
--senticnet-resource-path
--min-implicit-vote-sources
--target-implicit-ratio
--episodic-max-aspect-share
--disable-second-aspect-extraction
--seed
```

### Example runs

Dry-run without writing files:

```powershell
python dataset_builder\code\build_dataset.py --dry-run
```

Tune confidence and split ratios:

```powershell
python dataset_builder\code\build_dataset.py --confidence-threshold 0.45 --split-ratios 0.7,0.15,0.15
```

Force more open-aspect retention for domain-agnostic experiments:

```powershell
python dataset_builder\code\build_dataset.py --prefer-open-aspect --domain-agnostic-mode always
```

Run only against a specific source file folder:

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

## Outputs

### Review-level output

Files:

- `output/reviewlevel/train.jsonl`
- `output/reviewlevel/val.jsonl`
- `output/reviewlevel/test.jsonl`

Each row contains fields like:

- `id`
- `review_text`
- `domain`
- `source`
- `split`
- `labels`
- `target_text`

`target_text` is formatted as:

```text
aspect | sentiment | evidence ;; aspect | sentiment | evidence
```

### Episodic output

Files:

- `output/episodic/train.jsonl`
- `output/episodic/val.jsonl`
- `output/episodic/test.jsonl`
- `output/episodic/train_balanced.jsonl`

Each row contains fields like:

- `example_id`
- `parent_review_id`
- `review_text`
- `evidence_sentence`
- `domain`
- `aspect`
- `implicit_aspect`
- `sentiment`
- `label_type`
- `split`

### Reports

The builder writes diagnostics to `output/reports/`:

- `build_report.json`
- `data_quality_report.json`
- `episode_readiness_report.json`
- `normalization_report.json`
- `label_issue_candidates.jsonl`
- `skipped_rows.jsonl`

These are the fastest way to validate whether a newly ingested dataset is healthy enough for training.

## Optional Resources and Env Vars

SenticNet support is configured through:

- `resources/senticnet_seed.json`
- `--senticnet-resource-path`

The config layer also defines optional provider settings for LLM-backed helpers:

- `DEFAULT_LLM_PROVIDER`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `GROQ_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`

Those keys are optional for the default local build flow.



## Hand-off to ProtoNet

The usual next step is to copy the episodic output into `protonet/input/episodic/` and train the standalone ProtoNet module:

```powershell
Copy-Item "dataset_builder\output\episodic\*" -Destination "protonet\input\episodic\" -Recurse -Force
```

If you are already inside `dataset_builder/`, return to the repo root first or use relative paths carefully before copying outputs into `protonet/`.
