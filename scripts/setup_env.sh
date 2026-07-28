#!/usr/bin/env bash
# Create/refresh the Python environment.
#   CUDA_TAG=cu132 bash scripts/setup_env.sh   # CUDA build (default cu126)
#   CUDA_TAG=cpu   bash scripts/setup_env.sh   # CPU-only build
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CUDA_TAG="${CUDA_TAG:-cu126}"

echo "[setup] python: $($PYTHON --version 2>&1)"
if [ "$CUDA_TAG" = "cpu" ]; then
  "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  "$PYTHON" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi
"$PYTHON" -m pip install -r requirements.txt

"$PYTHON" - <<'EOF'
import torch, transformers
print(f"[setup] torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[setup] transformers {transformers.__version__}")
print("[setup] OK")
EOF
