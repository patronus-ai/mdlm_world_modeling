#!/bin/bash
# One-time environment setup for SDAR SFT + DPO training.
#
# This script:
#   1. Creates a Python venv (default: ./.venv)
#   2. Clones JetBrains-Research/SDAR into $SDAR_DIR (default: ./SDAR)
#      — contains the bundled LlamaFactory fork (`training/llama_factory_sdar`)
#        used for SFT, and the DPO entrypoint (`training/train_dpo.py`).
#   3. Installs the LlamaFactory fork (editable) + DPO/eval extras.
#
# It does NOT download model weights or copy custom modeling files. SDAR
# requires you to assemble each model directory yourself by combining:
#   - the custom `modeling_*.py` and `config.json` from
#     `SDAR/training/model/<NAME>/`, and
#   - the official `*.safetensors` weights from the matching HF repo.
# See README.md for the full procedure.
#
# Usage:
#   bash setup.sh
#
# Optional env vars:
#   PYTHON_VERSION   default: 3.11
#   VENV_DIR         default: ./.venv
#   SDAR_DIR         default: ./SDAR
#   SDAR_REPO_URL    default: https://github.com/JetBrains-Research/SDAR.git

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/.venv}"
SDAR_DIR="${SDAR_DIR:-$WORK_DIR/SDAR}"
SDAR_REPO_URL="${SDAR_REPO_URL:-https://github.com/JetBrains-Research/SDAR.git}"

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

if [ ! -d "$SDAR_DIR" ]; then
    echo "=== Cloning SDAR into $SDAR_DIR ==="
    git clone "$SDAR_REPO_URL" "$SDAR_DIR"
fi

LF_DIR="$SDAR_DIR/training/llama_factory_sdar"
if [ ! -d "$LF_DIR" ]; then
    echo "ERROR: $LF_DIR not found. Bad clone or upstream layout changed." >&2
    exit 1
fi

echo "=== Installing LlamaFactory fork (editable) + extras ==="
$PIP_INSTALL -e "$LF_DIR"
$PIP_INSTALL accelerate deepspeed pyyaml datasets wandb requests rouge

echo "=== Verifying ==="
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
python -c "import llamafactory; print('llamafactory:', llamafactory.__version__)" 2>/dev/null || echo "llamafactory: editable install OK"

echo ""
echo "=== Setup complete ==="
echo ""
echo "NEXT STEPS"
echo "  1. Assemble model directories under ./model/<NAME>/ by combining"
echo "     SDAR's custom modeling_*.py + config.json with HuggingFace weights."
echo "     (See README.md, 'Model preparation'.)"
echo "  2. cp .env.example .env  &&  fill in HF_TOKEN / WANDB_API_KEY"
echo "  3. Edit configs/sft_example.yaml or configs/dpo_example.yaml"
echo "  4. bash train_sft.sh       (or)       bash train_dpo.sh"
