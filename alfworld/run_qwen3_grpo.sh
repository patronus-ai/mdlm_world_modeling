#!/bin/bash
# GRPO — Qwen/Qwen3-4B-Instruct-2507 (SFT checkpoint) on ALFWorld via SDAR WM.
#
# Pre-reqs:
#   1. lmdeploy serving SDAR on :30001
#   2. python wm_proxy.py on :30000
#   3. dataset at data/alfworld_rl.jsonl (from build_dataset.py with env-walk)
set -e
cd /workspace/user/rl_training/alfworld
source .venv/bin/activate
[ -f /workspace/user/.env ] && export $(grep -v '^#' /workspace/user/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface/}
export USE_HF=1
export WM_ENDPOINT=${WM_ENDPOINT:-http://localhost:30000}
export ALFWORLD_WM_GUARD=${ALFWORLD_WM_GUARD:-1}
export TRAJECTORY_LOG=${TRAJECTORY_LOG:-/tmp/alfworld_trajectories.jsonl}
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# Set MODEL_PATH to your SFT checkpoint dir (e.g. output/alfworld_qwen3_4b_sft/v0-.../checkpoint-300)
MODEL_PATH=${MODEL_PATH:?"set MODEL_PATH to your Qwen3-4B SFT checkpoint dir"}
DATA=${DATA:-data/alfworld_rl.jsonl}
RUN_NAME="alfworld_qwen3_4b_grpo_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
WANDB_PROJECT="alfworld" \
WANDB_RUN_NAME="$RUN_NAME" \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --template qwen3 \
    --external_plugins alfworld_plugin.py \
    --reward_funcs alfworld_reward \
    --multi_turn_scheduler alfworld_scheduler \
    --max_turns 30 \
    --completion_length_limit_scope per_round \
    --stop_words '<|im_end|>' '<|endoftext|>' \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 32768 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --torch_dtype bfloat16 \
    --dataset "$DATA" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 128 \
    --max_length 32768 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --max_grad_norm 0.5 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant \
    --gradient_checkpointing true \
    --save_steps 25 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 0.95 \
    --num_iterations 1 \
    --beta 0.1 \
    --loss_scale default \
    --log_completions true \
    --report_to wandb \
    --run_name "$RUN_NAME" \
    --output_dir output/alfworld_qwen3_4b_grpo
