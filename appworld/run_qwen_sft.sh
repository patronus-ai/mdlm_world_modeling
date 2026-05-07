#!/bin/bash
# SFT — Qwen3-4B on AppWorld ground truth trajectories
cd /workspace/user/trl_training
source .venv/bin/activate
export $(grep -v '^#' .env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA

MODEL_PATH=${MODEL_PATH:-/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
RUN_NAME="appworld_qwen3_sft_final_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=5 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="appworld_qwen3_sft_final_$(date +%Y%m%d_%H%M%S)"
swift sft \
    --model $MODEL_PATH \
    --template qwen3 \
    --dataset /workspace/user/trl_training/data/appworld_sft_gpt_agent_clean.jsonl \
    --torch_dtype bfloat16 \
    --tuner_type full \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-6 \
    --max_length 16384 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing true \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/appworld_qwen3_sft_final
