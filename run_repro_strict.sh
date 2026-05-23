#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
export REVIEWOP_ENV=test

python -m compileall -q protonet dataset_builder backend
python -m pytest -q protonet/tests
python -m pytest -q dataset_builder/test_dataset_hardening.py dataset_builder/test_spacy_pipeline_offline.py dataset_builder/test_label_equivalence_conflicts_tdd.py dataset_builder/test_memory_sidecar_consistency_tdd.py
python -m pytest -q backend

python -m protonet.cli verify-artifact --artifact-dir dataset_builder/output
python -m protonet.cli compare --artifact-dir dataset_builder/output --output-dir protonet/output/repro_strict_grouped --protocol grouped --split test
python -m protonet.cli compare --artifact-dir dataset_builder/output --output-dir protonet/output/repro_strict_domain --protocol domain_holdout --split test
python -m protonet.scripts.verify_run_contract protonet/output/repro_strict_grouped
python -m protonet.scripts.verify_run_contract protonet/output/repro_strict_domain

npm --prefix frontend ci
npm --prefix frontend run build

