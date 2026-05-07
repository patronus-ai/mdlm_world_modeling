#!/bin/bash
# SFT — LiquidAI/LFM2.5-1.2B-Instruct on credential workflow demos
cd /workspace/user/trl_training
source .venv/bin/activate
export $(grep -v '^#' .env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/

RUN_NAME="appworld_lfm25_sft_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift sft \
    --model LiquidAI/LFM2.5-1.2B-Instruct \
    --template chatml \
    --dataset /workspace/user/trl_training/data/appworld_sft_lfm25.jsonl \
    --torch_dtype bfloat16 \
    --tuner_type full \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/appworld_lfm25_sft
