~# Dataset Builder (ABSA + Few-Shot Episodic)

This tool ingests raw review files from multiple sources and builds two aligned datasets:

- `reviewlevel`: one record per review for ABSA/seq2seq/encoder models.
- `episodic`: one record per episode for ProtoNet/meta-learning tasks.

It supports explicit and implicit aspects, evidence fields, optional API-assisted quality improvements, and strict clean-first output regeneration.

## Input / Output Layout

- Input: `dataset_builder/input/raw/` (`.csv`, `.json`, `.jsonl`)
- Output:
  - `dataset_builder/output/reviewlevel/normal/{train,val,test}.jsonl`
  - `dataset_builder/output/reviewlevel/augmented/{train,val,test}.jsonl`
  - `dataset_builder/output/episodic/normal/{train,val,test}.jsonl`
  - `dataset_builder/output/episodic/augmented/{train,val,test}.jsonl`

No `metadata/` or `debug_samples/` folders are generated.

## How schema auto-detection works

1. Heuristic matching against likely column names (`text`, `title`, `rating`, `id`, `domain`, `split`, etc).
2. If mapping is ambiguous and API key is configured, optional LLM tie-breaker is used.
3. Pipeline continues with heuristics if API is unavailable or request fails.

## Augmentation strategy

Augmented set is built from normalized reviewlevel records:

- explicit -> implicit rewrites
- mixed explicit+implicit phrasing
- aspect/sentiment preservation checks
- duplicate/low-quality filtering

Every augmented record includes `is_augmented`, `augmentation_type`, `source_record_id`, `preserved_aspects`, and `preserved_sentiments`.

## API keys

Create `dataset_builder/.env` (or root `.env`) from `.env.example`:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `GROQ_API_KEY`
- `GROQ_BASE_URL`
- `GROQ_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `DEFAULT_LLM_PROVIDER`

Default behavior is API-preferred (`--use-api true`) with automatic heuristic fallback.

## Run

From repo root:

```powershell
python dataset_builder/code/main.py --input dataset_builder/input/raw --output output --mode all
```

Other common runs:

```powershell
python dataset_builder/code/main.py --mode reviewlevel
python dataset_builder/code/main.py --mode episodic
python dataset_builder/code/main.py --augment true
python dataset_builder/code/main.py --cross-domain true
```

Useful flags:

- `--clean-first true|false` (default `true`)
- `--use-api true|false` (default `true`)
- `--preserve-official-splits true|false`
- `--max-aspects 5`
- `--min-review-length 8`
- `--near-dup-threshold 0.9`
- `--n-way 5 --k-shot 5 --q-query 10`
- `--strict-quality-filter true|false`
- `--target-multi-aspect-min 2`
- `--target-implicit-ratio 0.2`
- `--max-canonical-share 0.45`
- `--hard-negative-k 2`
- `--implicit-query-only true|false`
- `--min-evidence-span-chars 5`
- `--aspect-definitions-enabled true|false`
- `--domain-family-implicit-targets electronics:0.2,telecom:0.2,ecommerce:0.2,mobility:0.2,healthcare:0.2,services:0.2`
- `--cross-domain-min-domains 2`
- `--fallback-episode-policy relax_implicit_query,reduced_shots,reduced_way`
- `--max-evidence-fallback-rate 0.15`
- `--episode-task-mix aspect_classification:0.4,implicit_aspect_inference:0.3,aspect_sentiment_classification:0.3`
- `--hard-negative-strategy static|data_driven|hybrid`

## Runtime quality gates

The builder prints acceptance gate signals at the end of each run:

- canonical OOV rate
- leakage rate
- sentence fallback evidence rate
- implicit family coverage and target misses

## Aspect Memory

Aspect Memory stores aspect-term decisions in JSONL so future dataset builds can reuse prior mappings without introducing a database dependency.

Files written by the memory layer:

- `output/reports/aspect_memory/aspect_memory_events.jsonl`
- `output/reports/aspect_memory/aspect_memory_promotions.jsonl`
- `output/reports/aspect_memory/aspect_memory_calibration.json`

Common modes:

- `--memory-mode off`: do not read or write memory
- `--memory-mode collect`: store candidates and evidence only
- `--memory-mode resolve`: resolve aspect terms from memory when confidence is high enough
- `--mode eval`: read existing `reviewlevel/normal/{train,val,test}.jsonl`, keep memory read-only, and write `output/reports/eval_report.json`
- `--freeze-memory-during-eval true`: block all memory writes while evaluating

Decision policies:

- `--decision-policy deterministic`
- `--decision-policy hybrid`
- `--decision-policy stochastic`

Example commands:

```powershell
python dataset_builder/code/main.py --mode all --decision-policy deterministic --memory-mode off --seed 42
python dataset_builder/code/main.py --mode all --decision-policy hybrid --memory-mode resolve --min-confidence-for-hard-map 0.75 --decision-temperature 0.4 --seed 42
python dataset_builder/code/main.py --mode all --decision-policy stochastic --decision-temperature 0.8 --seed 7
python dataset_builder/code/main.py --workflow two-track --mode all --seed 42
```

## Common failure cases

- No rows produced: schema text column not detected and LLM fallback unavailable.
- Weak aspect coverage: dataset language/domain mismatch with heuristics.
- Few/no episodes: insufficient per-label samples for chosen `n_way/k_shot/q_query`.
- Empty augmented output: strict quality checks filtered generated rewrites.
