# Dataset Builder

DAGR-PIPE v4 dataset builder for explicit and implicit review features.

## Layout

- `dataset_builder/input/` raw input files
- `dataset_builder/output/explicit/` explicit feature exports
- `dataset_builder/output/implicit/` implicit feature exports
- `dataset_builder/output/reports/` build reports, diagnostics, and research manifests

## Run

```powershell
python dataset_builder\code\build_dataset.py --input-dir dataset_builder\input --output-dir dataset_builder\output
```

Chunked prototyping:

```powershell
python dataset_builder\code\build_dataset.py --sample-size 100 --chunk-size 25 --chunk-offset 0 --preview
```

Chunked write run (produces files in `dataset_builder/output/`):

```powershell
python dataset_builder\code\build_dataset.py --sample-size 100 --chunk-size 25 --chunk-offset 0
```

Research mode examples:

```powershell
python dataset_builder\code\build_dataset.py --implicit-mode zeroshot --multilingual-mode shared_vocab --use-coref
python dataset_builder\code\build_dataset.py --emit-review-set --review-set-size 300
python dataset_builder\code\build_dataset.py --gold-annotations-path dataset_builder\input\gold_annotations.jsonl
python dataset_builder\code\build_dataset.py --evaluation-protocol loo --domain-holdout hotel
python dataset_builder\code\build_dataset.py --no-enable-llm-fallback
python dataset_builder\code\build_dataset.py --train-fallback-general-policy cap --train-fallback-general-cap-ratio 0.15
python dataset_builder\code\build_dataset.py --train-sentiment-balance-mode cap_neutral --train-neutral-cap-ratio 0.50
python dataset_builder\code\run_experiment.py --plan-only --implicit-mode hybrid --no-drop
python dataset_builder\code\run_experiment.py --execute-v4-sweep
python dataset_builder\code\run_experiment.py --execute-v4-sweep --sweep-implicit-min-tokens 6,8 --sweep-min-text-tokens 3,4
python dataset_builder\code\run_experiment.py --execute-v4-sweep --gold-annotations-path dataset_builder\input\gold_annotations.jsonl --gold-min-rows-for-promotion 600 --apply-best-defaults --no-enable-llm-fallback
```

## V4 Quality Sweep

- `--execute-v4-sweep` runs a bounded config sweep for fallback/review reduction using V4-Base settings.
- Sweep dimensions include `implicit_mode`, `confidence_threshold`, `llm_fallback_threshold`, `implicit_min_tokens`, and `min_text_tokens` (plus optional coref).
- Candidate outputs are written under `dataset_builder/output/runs/<run_id>/candidates/`.
- Ranked results are written to `dataset_builder/output/runs/<run_id>/v4_sweep_results.json`.
- Quality gates:
  - `fallback_only_rate <= 0.22`
  - `needs_review_rows <= 1800`
  - `generic_implicit_aspects == 0`
  - `rejected_implicit_aspects == 0`
  - `train_general_dominance_rate <= 0.20`
- `--apply-best-defaults` writes promoted defaults to `dataset_builder/code/runtime_defaults.json` only when a candidate meets all quality gates.
- `--no-enable-llm-fallback` disables fallback parsing branch explicitly.
- Gold-driven promotion requires `gold_eval.has_gold_labels=true` and `gold_eval.num_rows_with_gold >= --gold-min-rows-for-promotion`.
- Novelty gates are active in sweep outputs via `ablation_summary` and required ablation completeness.

## Gold Annotation Workflow

1. Generate a review set template:
   - `python dataset_builder\code\build_dataset.py --emit-review-set --review-set-size 300`
   - Output: `dataset_builder/output/reports/review_set_template.jsonl`
2. Annotate each row in JSONL using this schema:
   - `record_id`: dataset row id
   - `domain`: canonical domain
   - `text`: review text
   - `gold_labels`: list of `{aspect, sentiment, start, end}`
   - `annotator_id`: reviewer identifier
   - `review_status`: e.g. `approved`
3. Save annotations to:
   - `dataset_builder/input/gold_annotations.jsonl`
4. Run builder/sweep with `--gold-annotations-path` to merge gold labels and compute `gold_eval`.

## Additive Report Keys

- `grounded_prediction_rate`
- `ungrounded_non_general_count`
- `gold_eval` (`aspect_f1`, `sentiment_f1`, `span_overlap_f1`, `by_domain`)
- `train_general_rows_before_policy`
- `train_general_rows_after_policy`
- `train_general_policy_applied`
- `train_sentiment_before_balance`
- `train_sentiment_after_balance`
- `train_general_dominance_rate`
- `output_quality.review_reason_counts`
- `output_quality.fallback_branch_counts`
- `domain_generalization` (`evaluation_protocol`, `domains_seen`, `by_domain`, `leave_one_domain_out`, `heldout_domain_metrics`)
- `novelty_identity` (method identity proof block)

## Preview Safety

- `--preview` is equivalent to `--dry-run` for output behavior.
- `--preview`/`--dry-run` are non-destructive:
  - they do not reset existing output artifacts
  - they do not write exports/reports (`explicit/`, `implicit/`, `reports/`, `compat/`)
- To generate files, run without `--preview` and `--dry-run`.
- Default output path is `dataset_builder/output/` (unless `--output-dir` is provided).
