#!/usr/bin/env bash
# Create the training / inference environment.
#
#   bash scripts/setup/install.sh [VENV_DIR]          (default: .venv)
#
# Environment variables:
#   TORCH_INDEX   PyTorch wheel index (default: CUDA 13, required for Blackwell / GB300 GPUs;
#                 use https://download.pytorch.org/whl/cu128 for CUDA 12.8 drivers)
#   UV            path to uv (default: uv on PATH)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${1:-$ROOT/.venv}
UV=${UV:-uv}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}

"$UV" venv --python 3.11 "$VENV"
export VIRTUAL_ENV="$VENV"
"$UV" pip install --index-url "$TORCH_INDEX" "torch==2.11.0" "torchvision==0.26.0"
"$UV" pip install -r "$ROOT/requirements/gpu.txt" --constraint <(printf 'torch==2.11.0\ntorchvision==0.26.0\n')
# openpi-client pins numpy<2, which conflicts with the rest of the stack; its runtime deps are above.
"$UV" pip install --no-deps "$ROOT/packages/openpi-client"
"$UV" pip install --no-deps -e "$ROOT"
"$VENV/bin/python" "$ROOT/scripts/setup/patch_transformers.py"
echo "Environment ready: source $VENV/bin/activate"
