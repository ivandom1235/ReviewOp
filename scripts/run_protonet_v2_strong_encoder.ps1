python -m pip install sentence-transformers

python -m protonet.cli compare `
  --artifact-dir dataset_builder\output\run_openai_500_v2 `
  --output-dir protonet\output\compare_openai_500_v2_st `
  --encoder sentence-transformers `
  --model-name sentence-transformers/all-MiniLM-L6-v2 `
  --top-k 5 `
  --split test
