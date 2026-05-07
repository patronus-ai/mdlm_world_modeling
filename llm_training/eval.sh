#!/bin/bash
# Loss-based eval over one or more validation JSONL files using ms-swift.
# Reports cross-entropy / token accuracy (no generation). For generation
# metrics (exact-match, ROUGE, semantic match), use evaluate.py instead.
#
# Required env vars (or set in .env):
#   HF_TOKEN  (only if model is private)
#
# Knobs:
#   MODEL_DIR       Trained / merged checkpoint dir (or HF model id)
#   EVAL_FILES      Space-separated JSONL paths to evaluate
#   OUTPUT_BASE     Base output dir; one subdir per file
#   NUM_GPUS / MASTER_PORT / ATTN_IMPL / MAX_LEN / PACKING / DEEPSPEED
#
# Example:
#   MODEL_DIR=output/qwen3_run1_merged \
#   EVAL_FILES="data/eval_a.jsonl data/eval_b.jsonl" \
#     bash eval.sh

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

if [ -f "$WORK_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$WORK_DIR/.env"
    set +a
fi

VENV_DIR="${VENV_DIR:-$WORK_DIR/swift-venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODEL_DIR="${MODEL_DIR:-$WORK_DIR/output/sft_run}"
EVAL_FILES="${EVAL_FILES:-$WORK_DIR/data/test.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-$WORK_DIR/output/eval}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_LEN="${MAX_LEN:-8192}"
PACKING="${PACKING:-true}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
DEEPSPEED="${DEEPSPEED:-zero2}"

export HF_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
[ "$DEEPSPEED" != "none" ] && EXTRA_ARGS+=(--deepspeed "$DEEPSPEED")

for EVAL_FILE in $EVAL_FILES; do
    NAME="$(basename "$EVAL_FILE" .jsonl)"
    OUT="$OUTPUT_BASE/$NAME"
    mkdir -p "$OUT"

    echo "=========================================="
    echo "Evaluating: $EVAL_FILE  ->  $OUT"
    echo "=========================================="

    NPROC_PER_NODE="$NUM_GPUS" \
    python3 -m torch.distributed.run \
        --nproc_per_node "$NUM_GPUS" \
        --master_port "$MASTER_PORT" \
        "$WORK_DIR/ms-swift/swift/cli/sft.py" \
        --model "$MODEL_DIR" \
        --val_dataset "$EVAL_FILE" \
        --torch_dtype "$TORCH_DTYPE" \
        --attn_impl "$ATTN_IMPL" \
        --num_train_epochs 0 \
        --do_train false \
        --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
        --max_length "$MAX_LEN" \
        --packing "$PACKING" \
        --dataset_num_proc 16 \
        --packing_num_proc 16 \
        --dataloader_num_workers 4 \
        --dataloader_prefetch_factor 4 \
        --dataloader_persistent_workers true \
        --load_from_cache_file true \
        --output_dir "$OUT" \
        --report_to none \
        --run_name "eval_${NAME}" \
        "${EXTRA_ARGS[@]}"
    echo ""
done
