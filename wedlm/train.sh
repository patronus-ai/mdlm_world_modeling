#!/bin/bash
# Launch WeDLM SFT via accelerate.
# Reads tokens from .env (never hardcoded). Set NUM_GPUS / CONFIG / etc.
#
# Required env vars (or set in .env):
#   HF_TOKEN          (only if base model is gated/private)
#   WANDB_API_KEY     (only if use_wandb: true in the config)
#
# Knobs:
#   CONFIG       Path to YAML config (default: ./configs/example.yaml)
#   WEDLM_DIR    Path to WeDLM checkout (default: ./WeDLM)
#   VENV_DIR     Path to venv (default: ./.venv)
#   NUM_GPUS     GPUs to use (default: 8)
#   MIXED_PRECISION  bf16|fp16|no (default: bf16)
#
# Examples:
#   bash train.sh
#   CONFIG=configs/my_run.yaml NUM_GPUS=4 bash train.sh

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

if [ -f "$WORK_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$WORK_DIR/.env"
    set +a
fi

VENV_DIR="${VENV_DIR:-$WORK_DIR/.venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

WEDLM_DIR="${WEDLM_DIR:-$WORK_DIR/WeDLM}"
TRAIN_PY="$WEDLM_DIR/finetune/train.py"
if [ ! -f "$TRAIN_PY" ]; then
    echo "ERROR: $TRAIN_PY not found. Run setup.sh first or set WEDLM_DIR." >&2
    exit 1
fi

CONFIG="${CONFIG:-$WORK_DIR/configs/example.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTHONUNBUFFERED=1

echo "=== WeDLM SFT ==="
echo "  config : $CONFIG"
echo "  gpus   : $NUM_GPUS"
echo "  prec   : $MIXED_PRECISION"
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --multi_gpu \
        --num_processes "$NUM_GPUS" \
        --mixed_precision "$MIXED_PRECISION" \
        "$TRAIN_PY" --config "$CONFIG"
else
    python "$TRAIN_PY" --config "$CONFIG"
fi
