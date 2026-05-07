#!/bin/bash
# SFT — Qwen3-4B on ALFWorld expert trajectories (1 epoch)
cd /workspace/user/alfworld
source /workspace/user/trl_training/.venv/bin/activate
export $(grep -v '^#' /workspace/user/trl_training/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/
export USE_HF=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 41000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA

MODEL_PATH=/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
RUN_NAME="alfworld_qwen3_4b_sft_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift sft \
    --model $MODEL_PATH \
    --template qwen3 \
    --dataset /workspace/user/alfworld/data/alfworld_sft_large.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing true \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/alfworld_qwen3_4b_sft
