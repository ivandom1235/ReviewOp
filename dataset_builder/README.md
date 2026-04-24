# Dataset Builder

Clean parallel implementation for evidence-grounded ReviewOps benchmark construction.

The package is intentionally rooted directly at `dataset_builder/`.

## Status

This is the production-ready ReviewOps benchmark construction pipeline. It provides:

- strict row/schema contracts
- deterministic review IDs
- fail-closed group IDs
- grouped train/val/test splitting
- evidence span validation
- learned symptom-pattern mining and validation
- artifact export with release gates
- artifact validation and summary scripts

This builder implements a modular, domain-agnostic architecture with externalized configuration.

## Input Format

Place input datasets inside:

```text
dataset_builder/input/
```

The scripts currently support CSV and JSONL review files. Each row needs:

- `text` or `review_text`
- one real grouping field: `group_id`, `product_id`, `business_id`, or `entity_id`
- optional `review_id`
- optional `domain`
- optional `domain_family`

Rows without a real grouping field fail closed. The builder does not invent group IDs
from review text because that can hide split leakage.

Example JSONL row:

```json
{"review_id":"r1","product_id":"p1","domain":"electronics","domain_family":"retail","text":"Battery life is excellent."}
```

## Output Contract

By default, `build_benchmark.py` writes the artifact directory to:

```text
dataset_builder/output/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
  quality_report.json
  artifact.zip
```

You can override this with `--output-dir`.

Artifact layout:

```text
dataset_builder/output/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
  quality_report.json
  artifact.zip
```

`artifact.zip` contains `train.jsonl`, `val.jsonl`, `test.jsonl`,
`manifest.json`, and `quality_report.json`.

Every split must be non-empty. `validate_artifact.py` checks:

- required split files
- `manifest.json`
- `quality_report.json`
- non-empty train/val/test splits
- unresolved `group_id`
- empty review text
- non-empty `gold_interpretations`
- evidence span validity
- grouped split leakage
- exact text leakage
- quality-report count consistency

## Commands

Run from the repository root.

Profile a dataset:

```powershell
python dataset_builder\scripts\profile_dataset.py dataset_builder\input\reviews.jsonl
```

Build a benchmark artifact from the default input folder:

```powershell
python dataset_builder\scripts\build_benchmark.py
```

`build_benchmark.py` defaults to:

- input: `dataset_builder\input`
- output: `dataset_builder\output`
- LLM provider: `none`

When the input path is a folder, the builder uses the first `.jsonl` or `.csv`
files in that folder, sorted by filename.

Build with an explicit input file:

```powershell
python dataset_builder\scripts\build_benchmark.py dataset_builder\input\reviews.jsonl
```

Build with an explicit output folder:

```powershell
python dataset_builder\scripts\build_benchmark.py dataset_builder\input\reviews.jsonl --output-dir dataset_builder\output
```

Build while selecting an LLM provider option:

```powershell
python dataset_builder\scripts\build_benchmark.py dataset_builder\input --llm openai --llm-model gpt-5-nano --overwrite
```

Build a deterministic sample:

```powershell
python dataset_builder\scripts\build_benchmark.py --sample-size 500 --seed 42 --overwrite
```

Build one deterministic chunk after sampling:

```powershell
python dataset_builder\scripts\build_benchmark.py --sample-size 5000 --chunk-size 1000 --chunk-offset 2000 --seed 42 --overwrite
```

Validate configuration and release gates without writing artifacts:

```powershell
python dataset_builder\scripts\build_benchmark.py --sample-size 100 --dry-run
```

Validate an artifact:

```powershell
python dataset_builder\scripts\validate_artifact.py dataset_builder\output
```

Summarize an artifact:

```powershell
python dataset_builder\scripts\summarize_artifact.py dataset_builder\output
```

Module invocation also works:

```powershell
python -m dataset_builder.scripts.profile_dataset dataset_builder\input\reviews.jsonl
python -m dataset_builder.scripts.build_benchmark
python -m dataset_builder.scripts.build_benchmark dataset_builder\input\reviews.jsonl --llm openai --output-dir dataset_builder\output --overwrite
python -m dataset_builder.scripts.validate_artifact dataset_builder\output
python -m dataset_builder.scripts.summarize_artifact dataset_builder\output
```

### CLI Options

`build_benchmark.py` supports:

```text
input                     Optional file or folder. Defaults to dataset_builder/input.
--output-dir PATH          Output artifact folder. Defaults to dataset_builder/output.
--llm {none,openai}        LLM provider selection. Defaults to none.
--llm-model MODEL          OpenAI model used when --llm openai. Defaults to gpt-5-nano.
--sample-size N            Deterministically keep N reviews after seeded shuffle.
--chunk-size N             Keep N reviews from the sampled/shuffled working set.
--chunk-offset N           Start offset for --chunk-size. Defaults to 0.
--seed N                   Seed for sampling and grouped splitting. Defaults to 42.
--train-ratio FLOAT        Train split ratio. Defaults to 0.8.
--val-ratio FLOAT          Validation split ratio. Defaults to 0.1.
--test-ratio FLOAT         Test split ratio. Defaults to 0.1.
--dry-run                  Run validation and release gates without writing output.
--overwrite                Replace a non-empty output directory.
```

When `--llm openai` is used, `OPENAI_API_KEY` must be present in the shell
environment. The builder calls the OpenAI Responses API as a verifier and records the
model, action, and confidence in row provenance. It does not silently fall back
to `none` if the key is missing.

`--sample-size`, `--chunk-size`, and `--chunk-offset` are applied before
benchmark conversion and splitting. The builder first shuffles deterministically
with `--seed`, then applies `--sample-size`, then applies the chunk window.

For normal builds, the builder fails if the output directory already contains files.
Pass `--overwrite` when intentionally replacing `dataset_builder\output`.

## Smoke Example

Create `dataset_builder\input\reviews.jsonl` with at least ten rows and
distinct `product_id` values, then run:

```powershell
python dataset_builder\scripts\profile_dataset.py dataset_builder\input\reviews.jsonl
python dataset_builder\scripts\build_benchmark.py --llm openai --llm-model gpt-5-nano --overwrite
python dataset_builder\scripts\validate_artifact.py dataset_builder\output
python dataset_builder\scripts\summarize_artifact.py dataset_builder\output
```

At least ten rows are recommended because the strict release gate requires
non-empty train, validation, and test splits.

## Learned Symptom Patterns

The builder does not use a fixed `SYMPTOM_PATTERNS` aspect mapping. Instead:

- `implicit/symptom_miner.py` mines repeated domain-neutral symptom phrases.
- `implicit/symptom_validator.py` promotes or queues candidates using support,
  evidence-valid rate, precision estimate, and domain scope.
- `implicit/symptom_store.py` stores promoted patterns for runtime matching.
- `implicit/symptom_rules.py` extracts candidates only from a supplied learned
  `SymptomPatternStore`.

Domain-scoped patterns only apply to reviews from domains where the pattern was
learned. Global patterns can apply across domains.

## Requirements

The package depends on several modern NLP and LLM libraries:

- `openai`: LLM-based verification and refinement
- `spacy`: Open-domain explicit extraction and linguistic parsing
- `rich`: High-fidelity CLI progress and reporting
- `python-dotenv`: Environment variable management for API keys

Install dependencies:

```powershell
pip install -r dataset_builder/requirements.txt
```

Also ensures the `en_core_web_sm` model is available for spaCy:

```powershell
python -m spacy download en_core_web_sm
```

## Tests

Run all tests:

```powershell
python -m unittest discover -s dataset_builder\tests -p "test_*.py" -v
```

Compile check:

```powershell
python -m compileall dataset_builder
```

Current focused coverage includes:

- core contracts and release gates
- learned symptom mining/validation/runtime matching
- candidate policy behavior
- artifact validation failure modes

## Notes

- `dataset_builder` implements the full paper-aligned pipeline across Stages A-H.
- The system is designed to be domain-agnostic at the model level, with domain-specific knowledge externalized in `dataset_builder/config/domains/`.
- All interpretations are grounded in evidence text with verified spans before export.
