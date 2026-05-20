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
python -m pytest -q dataset_builder/test_dataset_hardening.py
python -m protonet.cli --help
python -m protonet.scripts.run_research_program --help
python -m protonet.cli verify-artifact --artifact-dir dataset_builder/output --allow-missing-active-contract
python -m protonet.cli compare --artifact-dir dataset_builder/output --output-dir protonet/output/repro_grouped --protocol grouped --split test --allow-missing-active-contract --allow-failed-artifact
python -m protonet.cli compare --artifact-dir dataset_builder/output --output-dir protonet/output/repro_domain --protocol domain_holdout --split test --allow-missing-active-contract --allow-failed-artifact
python -m protonet.scripts.verify_run_contract protonet/output/repro_grouped
python -m protonet.scripts.verify_run_contract protonet/output/repro_domain

npm --prefix frontend ci
npm --prefix frontend run build
