#!/bin/bash
# One-time environment setup for ms-swift LoRA SFT.
# Prerequisites: NVIDIA GPU with a recent CUDA driver, Python 3.12, uv installed.
#
# Usage:
#   bash setup.sh
#
# Optional overrides (env vars):
#   PYTHON_VERSION  Python version for the venv (default: 3.12)
#   VENV_DIR        Path to the virtualenv (default: ./swift-venv)
#   SWIFT_DIR       Path to ms-swift checkout (default: ./ms-swift)
#   FLASH_ATTN_WHL  URL of a prebuilt flash-attn wheel matching your
#                   CUDA / torch / Python combo. If unset, flash-attn is
#                   skipped — set --attn_impl eager in train.sh instead.

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/swift-venv}"
SWIFT_DIR="${SWIFT_DIR:-$WORK_DIR/ms-swift}"

echo "=== Creating venv at $VENV_DIR (python $PYTHON_VERSION) ==="
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ ! -d "$SWIFT_DIR" ]; then
    echo "=== Cloning ms-swift into $SWIFT_DIR ==="
    git clone https://github.com/modelscope/ms-swift.git "$SWIFT_DIR"
fi

echo "=== Installing ms-swift + core dependencies ==="
uv pip install -e "$SWIFT_DIR"
uv pip install datasets accelerate deepspeed wandb requests rouge

if [ -n "${FLASH_ATTN_WHL:-}" ]; then
    echo "=== Installing flash-attn from $FLASH_ATTN_WHL ==="
    uv pip install "$FLASH_ATTN_WHL"
fi

echo "=== Verifying installation ==="
python3 -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
python3 -c "import swift; print('ms-swift:', swift.__version__)"
python3 -c "
try:
    import flash_attn; print('flash_attn:', flash_attn.__version__)
except ImportError:
    print('flash_attn: not installed (use --attn_impl eager)')
"

echo ""
echo "=== Setup complete ==="
echo "Next: copy .env.example -> .env, fill in tokens, then 'bash train.sh'"
