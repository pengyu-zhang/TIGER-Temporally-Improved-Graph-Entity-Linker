#!/usr/bin/env bash
# Full experiment matrix (paper main table): for each config and each
# training-set variant, train on every year in YEARS and evaluate on all
# of them. Finished runs are skipped, so the script is resumable.
#
#   bash scripts/run_all.sh                      # all three configs
#   CONFIGS=configs/default.yaml bash scripts/run_all.sh
#   SEEDS="52313 1 2" CONFIGS=configs/default.yaml bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
YEARS="${YEARS:-2019,2020,2021,2022}"
VARIANTS="${VARIANTS:-new_entities continual_entities}"
CONFIGS="${CONFIGS:-configs/baseline.yaml configs/paper.yaml configs/default.yaml}"
SEEDS="${SEEDS:-}"   # empty = config seed only

IFS=',' read -ra YEAR_LIST <<< "$YEARS"

for CONFIG in $CONFIGS; do
  RUN_NAME=$(basename "$CONFIG" .yaml)
  for VARIANT in $VARIANTS; do
    for YEAR in "${YEAR_LIST[@]}"; do
      for SEED in ${SEEDS:-config}; do
        if [ "$SEED" = "config" ]; then
          OUT="outputs/$RUN_NAME/$VARIANT/$YEAR"
          SEED_ARGS=()
        else
          OUT="outputs/$RUN_NAME/$VARIANT/$YEAR/seed$SEED"
          SEED_ARGS=(--set "seed=$SEED")
        fi
        echo "=== [run_all] $RUN_NAME $VARIANT $YEAR seed=$SEED ==="
        PYTHONPATH=src "$PYTHON" -m tiger.train \
          --config "$CONFIG" --variant "$VARIANT" --year "$YEAR" \
          --output-dir "$OUT" --skip-existing "${SEED_ARGS[@]}"
        if [ ! -f "$OUT/results_test.jsonl" ] || \
           ! grep -q "\"test_year\": ${YEAR_LIST[-1]}" "$OUT/results_test.jsonl"; then
          PYTHONPATH=src "$PYTHON" -m tiger.evaluate \
            --config "$CONFIG" --model-dir "$OUT" --test-years "$YEARS" "${SEED_ARGS[@]}"
        fi
      done
    done
  done
done

echo "[run_all] complete"
