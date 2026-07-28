#!/usr/bin/env bash
# Evaluate a trained model on one or more test years.
#   bash scripts/evaluate.sh [config] [model_dir] [test_years]
# Defaults: configs/default.yaml outputs/default/new_entities/2019 2019,2020,2021,2022
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${1:-configs/default.yaml}"
MODEL_DIR="${2:-outputs/default/new_entities/2019}"
TEST_YEARS="${3:-2019,2020,2021,2022}"

PYTHONPATH=src "$PYTHON" -m tiger.evaluate \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --test-years "$TEST_YEARS"
