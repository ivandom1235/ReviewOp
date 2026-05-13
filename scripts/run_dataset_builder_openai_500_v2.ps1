python dataset_builder\scripts\build_benchmark.py dataset_builder\input `
  --output-dir dataset_builder\output\run_openai_500_v2 `
  --sample-size 500 `
  --profile stability `
  --llm openai `
  --llm-model gpt-4.1-mini `
  --max-workers 4 `
  --seed 42 `
  --domain-mode full `
  --provisional-policy strict `
  --aspect-memory-auto-promote `
  --domain-holdout-domain laptop `
  --overwrite

python dataset_builder\scripts\verify_artifact.py `
  dataset_builder\output\run_openai_500_v2 `
  --profile stability `
  --expected-rows 500 `
  --require-organic-memory `
  --require-rejected-row-audit `
  --require-counterfactual-source-breakdown

python dataset_builder_patches\finalize_artifact_status.py dataset_builder\output\run_openai_500_v2

python dataset_builder_patches\strict_novel_memory_audit.py dataset_builder\output\run_openai_500_v2

python dataset_builder_patches\write_active_artifact_contract.py `
  --artifact-dir dataset_builder\output\run_openai_500_v2 `
  --output CURRENT_ACTIVE_ARTIFACT.json
