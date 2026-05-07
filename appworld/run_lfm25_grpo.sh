#!/bin/bash
# GRPO — LiquidAI/LFM2.5-1.2B-Instruct on AppWorld RL tasks with WM
cd /workspace/user/trl_training
source .venv/bin/activate
export $(grep -v '^#' .env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/
export WM_ENDPOINT=http://localhost:30000/predict
export RUNPOD_ENDPOINT_ID=""
export TRAJECTORY_LOG=/tmp/appworld_lfm25_trajectories.jsonl

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA

MODEL_PATH=LiquidAI/LFM2.5-1.2B-Instruct
RUN_NAME="appworld_lfm25_grpo_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --template chatml \
    --external_plugins appworld_plugin.py \
    --reward_funcs appworld_reward \
    --multi_turn_scheduler appworld_scheduler \
    --max_turns 10 \
    --completion_length_limit_scope per_round \
    --stop_words '<|im_end|>' '<|endoftext|>' \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --torch_dtype bfloat16 \
    --dataset /workspace/user/trl_training/data/appworld_rl_split_clean.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 512 \
    --max_length 16384 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.5 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant \
    --gradient_checkpointing true \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 0.95 \
    --num_iterations 1 \
    --beta 0.1 \
    --loss_scale default \
    --log_completions true \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/appworld_lfm25_grpo
