#!/bin/bash
# SFT — LiquidAI/LFM2.5-1.2B-Instruct on ALFWorld expert trajectories.
set -e
cd /workspace/user/rl_training/alfworld
source .venv/bin/activate
[ -f /workspace/user/.env ] && export $(grep -v '^#' /workspace/user/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface/}
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export USE_HF=1   # ms-swift defaults to ModelScope; force HuggingFace

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

MODEL_PATH=${MODEL_PATH:-/workspace/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/6314d2b7cf28a6ae9de9d3e77dcfcd9c9f281c77}
DATA=${DATA:-data/alfworld_sft.jsonl}
RUN_NAME="alfworld_lfm25_sft_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
WANDB_PROJECT="alfworld" \
WANDB_RUN_NAME="$RUN_NAME" \
swift sft \
    --model "$MODEL_PATH" \
    --template chatml \
    --dataset "$DATA" \
    --torch_dtype bfloat16 \
    --tuner_type full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
    --max_length 16384 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing true \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/alfworld_lfm25_sft
