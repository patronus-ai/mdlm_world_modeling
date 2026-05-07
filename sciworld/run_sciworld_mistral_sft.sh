#!/bin/bash
# SFT — Mistral-7B on GPT-5.5 ScienceWorld demonstrations (1 epoch)
cd /workspace/user/sciworld_training
source /workspace/user/trl_training/.venv/bin/activate
export $(grep -v '^#' /workspace/user/trl_training/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA

MODEL_PATH=/workspace/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71
RUN_NAME="sciworld_mistral_sft_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift sft \
    --model $MODEL_PATH \
    --template llama \
    --dataset /workspace/user/sciworld_training/data/sciworld_sft_gpt_perfect_mistral.jsonl \
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
    --save_steps 5 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/sciworld_mistral_sft
