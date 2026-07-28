#!/usr/bin/env bash
# End-to-end smoke test on a tiny slice of the real data: trains each
# config on 200 samples, evaluates on 500 mentions against a 2,000-entity
# pool, and checks that recall@64 is produced. Finishes in minutes.
# Requires data/processed/graph_tempel for year 2019 (scripts/prepare_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
YEAR="${YEAR:-2019}"
VARIANT="${VARIANT:-new_entities}"

for CONFIG in configs/baseline.yaml configs/paper.yaml configs/default.yaml; do
  RUN_NAME=$(basename "$CONFIG" .yaml)
  OUT="outputs/smoke/$RUN_NAME"
  echo "=== [smoke] $RUN_NAME: train ==="
  PYTHONPATH=src "$PYTHON" -m tiger.train \
    --config "$CONFIG" --variant "$VARIANT" --year "$YEAR" \
    --output-dir "$OUT" \
    --max-train-samples 200 --max-valid-samples 200

  echo "=== [smoke] $RUN_NAME: evaluate ==="
  PYTHONPATH=src "$PYTHON" -m tiger.evaluate \
    --config "$CONFIG" --model-dir "$OUT" --test-years "$YEAR" \
    --max-eval-samples 500 --max-pool 2000

  grep -q "recall@64" "$OUT/results_test.jsonl" \
    || { echo "[smoke] FAIL: no recall@64 in $OUT/results_test.jsonl"; exit 1; }
  echo "=== [smoke] $RUN_NAME OK ==="
done

echo "[smoke] all configs passed"
