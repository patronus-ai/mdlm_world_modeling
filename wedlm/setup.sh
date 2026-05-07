#!/bin/bash
# One-time environment setup for WeDLM SFT.
#
# This script:
#   1. Creates a Python venv (default: ./.venv)
#   2. Clones tencent/WeDLM into $WEDLM_DIR (default: ./WeDLM)
#   3. Installs WeDLM + the finetune extras (accelerate, deepspeed, datasets, pyyaml)
#
# It does NOT install MagiAttention. MagiAttention is an optional but
# recommended attention backend whose installation is non-trivial (CUDA
# kernel compilation). Install it yourself by following the upstream guide:
#
#   https://github.com/SandAI-org/MagiAttention
#
# Then set `attention_backend: "magi"` in your config. If you skip
# MagiAttention, set `attention_backend: "dense"` and training still works.
#
# Usage:
#   bash setup.sh
#
# Optional env vars:
#   PYTHON_VERSION   default: 3.11
#   VENV_DIR         default: ./.venv
#   WEDLM_DIR        default: ./WeDLM

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/.venv}"
WEDLM_DIR="${WEDLM_DIR:-$WORK_DIR/WeDLM}"

echo "=== Creating venv at $VENV_DIR (python $PYTHON_VERSION) ==="
if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
    python"$PYTHON_VERSION" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PIP_INSTALL="pip install"
if command -v uv >/dev/null 2>&1; then
    PIP_INSTALL="uv pip install"
fi

if [ ! -d "$WEDLM_DIR" ]; then
    echo "=== Cloning WeDLM into $WEDLM_DIR ==="
    git clone https://github.com/Tencent/WeDLM.git "$WEDLM_DIR"
fi

echo "=== Installing WeDLM (editable) ==="
$PIP_INSTALL -e "$WEDLM_DIR"

echo "=== Installing finetune extras ==="
$PIP_INSTALL accelerate deepspeed pyyaml datasets json-repair wandb requests rouge

echo ""
echo "=== Setup complete ==="
echo ""
echo "NEXT STEPS"
echo "  1. (Optional, recommended) Install MagiAttention:"
echo "       https://github.com/SandAI-org/MagiAttention"
echo "     Then set 'attention_backend: \"magi\"' in your config."
echo "     Otherwise set 'attention_backend: \"dense\"'."
echo "  2. cp .env.example .env  &&  fill in HF_TOKEN / WANDB_API_KEY"
echo "  3. Edit configs/example.yaml  (set model_path, train_data, output_dir)"
echo "  4. bash train.sh"
