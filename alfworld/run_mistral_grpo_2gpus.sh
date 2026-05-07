#!/bin/bash
# GRPO — Mistral-7B (SFT checkpoint), 2-GPU via torchrun + zero2.
#
# 7B model needs 2 GPUs: each holds full weights (~14GB bf16), but
# Adam states + gradients are partitioned (zero2). Both run vLLM colocate.
# Longer context (32768) for multi-turn ALFWorld episodes.
#
# Pre-reqs:
#   1. lmdeploy serving SDAR on :30001
#   2. python wm_proxy.py on :30000
#   3. data/alfworld_rl_mistral.jsonl
#   4. ALFWorld env server on :30003 (for reward only — not used during rollout)
set -e
cd /workspace/user/alfworld
source /workspace/user/trl_training/.venv/bin/activate
[ -f /workspace/user/trl_training/.env ] && export $(grep -v '^#' /workspace/user/trl_training/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface/}
export USE_HF=1
export WM_ENDPOINT=${WM_ENDPOINT:-http://localhost:30000}
export ALFWORLD_WM_GUARD=1
export TRAJECTORY_LOG=${TRAJECTORY_LOG:-/tmp/alfworld_mistral_trajectories.jsonl}
export PYTHONPATH=/workspace/user/alfworld:$PYTHONPATH
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset HOST_NODE_ADDR NODE_ADDR NCCL_IB_HCA NUM_NODES NODE_ROLE WORLD_SIZE

MODEL_PATH=${MODEL_PATH:-output/alfworld_mistral_sft/v1-20260501-062621/checkpoint-350}
DATA=${DATA:-data/alfworld_rl_mistral.jsonl}
RUN_NAME="alfworld_mistral_grpo_2gpu_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --template llama \
    --external_plugins alfworld_plugin.py \
    --reward_funcs alfworld_reward \
    --multi_turn_scheduler alfworld_scheduler \
    --max_turns 30 \
    --completion_length_limit_scope per_round \
    --stop_words '</s>' \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 32768 \
    --sleep_level 1 \
    --deepspeed zero2 \
    --offload_model false \
    --offload_optimizer false \
    --torch_dtype bfloat16 \
    --dataset "$DATA" \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 256 \
    --max_length 32768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --max_grad_norm 0.5 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant \
    --gradient_checkpointing true \
    --save_steps 5 \
    --save_total_limit 10 \
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
    --output_dir output/alfworld_mistral_grpo
