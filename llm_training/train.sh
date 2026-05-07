#!/bin/bash
# Full fine-tuning SFT launcher for ms-swift.
# Configure via .env or env vars; do NOT hardcode tokens here.
#
# Required env vars (set in .env or your shell):
#   HF_TOKEN          HuggingFace token (set "" if model is public)
#   WANDB_API_KEY     wandb key (or set REPORT_TO=none to disable)
#
# Common knobs (all have defaults):
#   MODEL                       e.g. Qwen/Qwen3-8B
#   USE_HF                      true|false (true = HuggingFace, false = ModelScope)
#   TEMPLATE                    optional chat template override (e.g. chatml)
#   TRAIN_FILE / VAL_FILE       JSONL files with {"messages": [...]} per line
#   OUTPUT_DIR                  where checkpoints go
#   RUN_NAME                    wandb run name
#   NUM_GPUS                    GPUs to use (default 8)
#   MASTER_PORT                 torch.distributed master port (default 29500)
#   ATTN_IMPL                   flash_attn|eager|sdpa
#   PER_DEVICE_TRAIN_BATCH_SIZE / GRAD_ACCUM_STEPS / LR / EPOCHS / MAX_LEN
#   PACKING                     true|false (sequence packing)
#   DEEPSPEED                   zero2|zero3|none
#   REPORT_TO                   wandb|none
#
# Example:
#   MODEL=Qwen/Qwen3-8B TRAIN_FILE=data/train.jsonl VAL_FILE=data/test.jsonl \
#     RUN_NAME=qwen3_run1 bash train.sh

set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# ── load secrets / config from .env if present ───────────────────────────────
if [ -f "$WORK_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$WORK_DIR/.env"
    set +a
fi

# ── activate venv ────────────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$WORK_DIR/swift-venv}"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── config defaults ──────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-8B}"
USE_HF="${USE_HF:-true}"
TEMPLATE="${TEMPLATE:-}"

TRAIN_FILE="${TRAIN_FILE:-$WORK_DIR/data/train.jsonl}"
VAL_FILE="${VAL_FILE:-$WORK_DIR/data/test.jsonl}"

RUN_NAME="${RUN_NAME:-sft_run}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/output/$RUN_NAME}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

EPOCHS="${EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHED="${LR_SCHED:-cosine}"
MAX_LEN="${MAX_LEN:-8192}"
PACKING="${PACKING:-true}"

EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

DEEPSPEED="${DEEPSPEED:-zero2}"
REPORT_TO="${REPORT_TO:-wandb}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

# ── build optional flags ─────────────────────────────────────────────────────
EXTRA_ARGS=()
[ -n "$TEMPLATE" ] && EXTRA_ARGS+=(--template "$TEMPLATE")
[ "$USE_HF" = "true" ] && EXTRA_ARGS+=(--use_hf true)
[ "$DEEPSPEED" != "none" ] && EXTRA_ARGS+=(--deepspeed "$DEEPSPEED")

mkdir -p "$OUTPUT_DIR"

echo "=== ms-swift Full Fine-Tuning SFT ==="
echo "  model  : $MODEL"
echo "  train  : $TRAIN_FILE"
echo "  val    : $VAL_FILE"
echo "  output : $OUTPUT_DIR"
echo "  gpus   : $NUM_GPUS  ($CUDA_VISIBLE_DEVICES)"
echo "  attn   : $ATTN_IMPL"
echo ""

NPROC_PER_NODE="$NUM_GPUS" \
python3 -m torch.distributed.run \
    --nproc_per_node "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    "$WORK_DIR/ms-swift/swift/cli/sft.py" \
    --model "$MODEL" \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --tuner_type full \
    --torch_dtype "$TORCH_DTYPE" \
    --attn_impl "$ATTN_IMPL" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHED" \
    --max_length "$MAX_LEN" \
    --packing "$PACKING" \
    --dataset_num_proc 16 \
    --packing_num_proc 16 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 4 \
    --dataloader_persistent_workers true \
    --load_from_cache_file true \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --gradient_checkpointing true \
    --output_dir "$OUTPUT_DIR" \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    "${EXTRA_ARGS[@]}"
