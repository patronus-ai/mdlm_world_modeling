#!/bin/bash
# SDAR full-finetune SFT via LlamaFactory.
# Reads tokens from .env (never hardcoded).
#
# Knobs:
#   CONFIG       Path to YAML config (default: ./configs/sft_example.yaml)
#   SDAR_DIR     Path to SDAR checkout (default: ./SDAR)
#   VENV_DIR     Path to venv (default: ./.venv)
#   NUM_GPUS     GPUs to use (default: 8)
#   MASTER_PORT  default: 12345
#
# Required env vars (or set in .env):
#   HF_TOKEN      only if model is gated
#   WANDB_API_KEY only if report_to: wandb in the config

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

SDAR_DIR="${SDAR_DIR:-$WORK_DIR/SDAR}"
LF_DIR="$SDAR_DIR/training/llama_factory_sdar"
LAUNCHER="$LF_DIR/src/llamafactory/launcher.py"
if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: $LAUNCHER not found. Run setup.sh first or set SDAR_DIR." >&2
    exit 1
fi

CONFIG="${CONFIG:-$WORK_DIR/configs/sft_example.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTHONUNBUFFERED=1

echo "=== SDAR SFT (LlamaFactory) ==="
echo "  config       : $CONFIG"
echo "  gpus         : $NUM_GPUS"
echo "  launcher     : $LAUNCHER"
echo ""

# LlamaFactory expects to run from its own root so relative paths inside
# the YAML (e.g. examples/deepspeed/ds_z3_config.json) resolve correctly.
cd "$LF_DIR"

torchrun \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr 127.0.0.1 \
    --master_port "$MASTER_PORT" \
    "$LAUNCHER" \
    "$CONFIG"
