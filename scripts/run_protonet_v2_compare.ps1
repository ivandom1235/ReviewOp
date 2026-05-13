python -m protonet.cli compare `
  --artifact-dir dataset_builder\output\run_openai_500_v2 `
  --output-dir protonet\output\compare_openai_500_v2 `
  --encoder hashing `
  --top-k 5 `
  --split test

python -m protonet.verify_run protonet\output\compare_openai_500_v2
