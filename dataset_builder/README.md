# Dataset Builder (V6)

Builds benchmark files used by ProtoNet training and evaluation.

## Command

Run from repo root:

```powershell
python dataset_builder\code\build_dataset.py [options]
```

## Quick Examples

```powershell
# Standard build
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output

# Small sampled preview run with OpenAI
python dataset_builder\code\build_dataset.py --run-profile debug --sample-size 100 --llm-provider openai --preview

# Zip existing artifacts only
python dataset_builder\code\build_dataset.py --zip-only --output-dir dataset_builder\output
```

## Recommended Core Flags

These are the flags most users should care about:

- `--input-dir`
- `--output-dir`
- `--run-profile` (`research` for full runs, `debug` for sampled runs)
- `--sample-size`
- `--chunk-size`
- `--chunk-offset`
- `--llm-provider`
- `--llm-model-name`
- `--preview`
- `--dry-run`
- `--zip-only`
- `--no-llm-cache`
- `--progress` / `--no-progress`

The rest of the CLI is advanced tuning. Keep using it only when you need to override pipeline behavior.

## Output Files

- `dataset_builder/output/benchmark/ambiguity_grounded/train.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/val.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/test.jsonl`
- `dataset_builder/output/benchmark/ambiguity_grounded/metadata.json`
- `dataset_builder/output/reports/build_report.json`

## Options

### Input, output, and run control

- `--input-dir` input dataset directory
- `--output-dir` output root directory
- `--seed` random seed (default: `42`)
- `--text-column` override detected text column
- `--sample-size` number of rows to keep after shuffle
- `--chunk-size` number of rows in chunked run
- `--chunk-offset` start offset for chunking (default: `0`)
- `--run-profile` choices: `research`, `debug` (default: `research`)
- `--artifact-mode` choices: `auto`, `debug_artifacts`, `research_release` (default: `auto`)
- `--debug-benchmark-max-rows` cap rows in debug artifact mode (default: `180`)
- `--dry-run` run pipeline in non-destructive mode
- `--preview` preview mode (also non-destructive)
- `--zip-only` skip build and zip existing benchmark/report artifacts

### LLM and recovery

- `--llm-provider` choices: `auto`, `openai`, `runpod`, `ollama`, `mock` (default: `auto`)
- `--llm-model-name` model name/id (reads `REVIEWOP_LLM_MODEL_NAME`, or the active provider's `RUNPOD_MODEL` / `CLAUDE_MODEL` / `GROQ_MODEL` / `OPENAI_MODEL` / `OLLAMA_MODEL`; fallback: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `--enable-reasoned-recovery` enable reasoned recovery
- `--no-enable-reasoned-recovery` disable reasoned recovery
- `--max-workers` max parallel workers (default: `10`)
- `--enable-llm-fallback` enable LLM fallback
- `--no-enable-llm-fallback` disable LLM fallback
- `--llm-fallback-threshold` fallback confidence threshold (default from runtime defaults, fallback: `0.65`)
- `--no-llm-cache`, `--no-cache-llm` bypass and clear LLM cache for this run
- `--discovery-mode` enable Open-Domain Discovery for novel aspects (default: `true`)
- `--no-discovery-mode` disable Open-Domain Discovery
- `--discovery-min-confidence` threshold for novel aspect promotion (default: `0.55`)
- `--discovery-stability-threshold` required occurrences for ontology promotion (default: `5`)

### Extraction and preprocessing

- `--confidence-threshold` base confidence threshold (default from runtime defaults, fallback: `0.6`)
- `--max-aspects` max aspects per row (default: `20`)
- `--min-text-tokens` min tokens for input text (default: `4`)
- `--implicit-min-tokens` min tokens for implicit readiness (default: `8`)
- `--implicit-mode` choices: `zeroshot`, `supervised`, `hybrid`, `heuristic`, `benchmark` (default from runtime defaults, fallback: `zeroshot`)
- `--multilingual-mode` multilingual strategy (default: `shared_vocab`)
- `--use-coref` enable coreference feature path
- `--no-use-coref` disable coreference feature path
- `--language-detection-mode` language detection mode (default: `heuristic`)
- `--no-drop` keep rows that would otherwise be dropped
- `--benchmark-key` benchmark key override
- `--model-family` model-family tag (default: `heuristic_latent`)
- `--augmentation-mode` augmentation mode (default: `none`)
- `--prompt-mode` prompt mode (default: `constrained`)
- `--gold-annotations-path` path to gold annotation file
- `--emit-review-set` emit review set template file
- `--review-set-size` review set size when emitted (default: `300`)

### Evaluation and split strategy

- `--evaluation-protocol` choices: `random`, `loo`, `source-free` (default: `random`)
- `--domain-holdout` holdout domain value
- `--enforce-grounding` enforce grounding checks
- `--no-enforce-grounding` disable grounding enforcement
- `--high-difficulty` enable high difficulty mode
- `--no-high-difficulty` disable high difficulty mode
- `--adversarial-refine` enable adversarial refine path
- `--no-adversarial-refine` disable adversarial refine path

### Domain conditioning and leakage controls

- `--no-domain-conditioning` disable domain conditioning
- `--no-strict-domain-conditioning` disable strict domain conditioning
- `--domain-conditioning-mode` choices: `adaptive_soft`, `strict_hard`, `off` (default: `adaptive_soft`)
- `--train-domain-conditioning-mode` choices: `adaptive_soft`, `strict_hard`, `off`
- `--eval-domain-conditioning-mode` choices: `adaptive_soft`, `strict_hard`, `off`
- `--domain-prior-boost` domain prior boost (default: `0.05`)
- `--domain-prior-penalty` domain prior penalty (default: `0.08`)
- `--weak-domain-support-row-threshold` weak support threshold (default: `80`)
- `--unseen-non-general-coverage-min` minimum unseen non-general coverage (default: `0.55`)
- `--unseen-implicit-not-ready-rate-max` max unseen implicit-not-ready rate (default: `0.35`)
- `--unseen-domain-leakage-row-rate-max` max unseen leakage row rate (default: `0.02`)

### Train filtering, balancing, and top-up

- `--train-fallback-general-policy` choices: `keep`, `cap`, `drop` (default: `cap`)
- `--train-fallback-general-cap-ratio` cap ratio for fallback-general rows (default: `0.15`)
- `--train-review-filter-mode` choices: `keep`, `drop_needs_review`, `reasoned_strict` (default: `reasoned_strict`)
- `--train-salvage-mode` choices: `off`, `recover_non_general` (default: `recover_non_general`)
- `--train-salvage-confidence-threshold` salvage confidence threshold (default: `0.56`)
- `--train-salvage-accepted-support-types` csv support types (default: `exact,near_exact,gold`)
- `--train-sentiment-balance-mode` choices: `none`, `cap_neutral`, `cap_neutral_with_negative_floor`, `cap_neutral_with_dual_floor` (default: `cap_neutral_with_dual_floor`)
- `--train-neutral-cap-ratio` neutral cap ratio (default: `0.5`)
- `--train-min-negative-ratio` minimum negative ratio (default: `0.12`)
- `--train-min-positive-ratio` minimum positive ratio (default: `0.12`)
- `--train-max-positive-ratio` maximum positive ratio (default: `0.5`)
- `--train-neutral-max-ratio` maximum neutral ratio (default: `0.58`)
- `--train-topup-recovery-mode` choices: `off`, `strict_topup` (default: `strict_topup`)
- `--train-topup-confidence-threshold` top-up confidence threshold (default: `0.58`)
- `--train-topup-staged-recovery` enable staged top-up recovery
- `--no-train-topup-staged-recovery` disable staged top-up recovery
- `--train-topup-stage-b-confidence-threshold` stage B confidence threshold (default: `0.54`)
- `--train-topup-allow-weak-support-in-stage-c` allow weak support in stage C
- `--no-train-topup-allow-weak-support-in-stage-c` disallow weak support in stage C
- `--train-topup-stage-c-confidence-threshold` stage C confidence threshold (default: `0.52`)
- `--train-topup-allowed-support-types` csv support types for top-up (default: `exact,near_exact,gold`)
- `--train-target-min-rows` minimum train rows target (default: `1600`)
- `--train-target-max-rows` maximum train rows target (default: `2000`)

### Strict implicit quality gates

- `--strict-implicit-enabled` enable strict quality checks
- `--no-strict-implicit-enabled` disable strict quality checks
- `--strict-review-sample-size` strict review sample size (default: `200`)
- `--strict-explicit-in-implicit-rate-max` max explicit-in-implicit rate (default: `0.0`)
- `--strict-boundary-fp-max` max boundary false positives (default: `0`)
- `--strict-h2-h3-ratio-min` min H2/H3 ratio (default: `0.35`)
- `--strict-multi-aspect-ratio-min` min multi-aspect ratio (default: `0.12`)
- `--strict-challenge-macro-f1-min` min strict challenge macro-F1 (default: `0.5`)

### Progress display

- `--progress` enable progress bar
- `--no-progress` disable progress bar

## Show Outputs

```powershell
Get-ChildItem dataset_builder\output\benchmark\ambiguity_grounded
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\train.jsonl -TotalCount 3
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\val.jsonl -TotalCount 3
Get-Content dataset_builder\output\benchmark\ambiguity_grounded\test.jsonl -TotalCount 3
```

## Next Step

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_grounded
```
