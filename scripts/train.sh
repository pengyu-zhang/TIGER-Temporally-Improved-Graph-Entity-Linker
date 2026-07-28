#!/usr/bin/env bash
# Train one model.
#   bash scripts/train.sh [config] [variant] [year] [output_dir]
# Defaults: configs/default.yaml new_entities 2019 outputs/<run>/<variant>/<year>
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CONFIG="${1:-configs/default.yaml}"
VARIANT="${2:-new_entities}"
YEAR="${3:-2019}"
RUN_NAME=$(basename "$CONFIG" .yaml)
OUTPUT="${4:-outputs/$RUN_NAME/$VARIANT/$YEAR}"

PYTHONPATH=src "$PYTHON" -m tiger.train \
  --config "$CONFIG" \
  --variant "$VARIANT" \
  --year "$YEAR" \
  --output-dir "$OUTPUT"
