#!/bin/bash
# GRPO — Mistral-7B on ScienceWorld tasks with WM (on top of SFT checkpoint)
cd /workspace/user/sciworld_training
source /workspace/user/trl_training/.venv/bin/activate
export $(grep -v '^#' /workspace/user/trl_training/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/
export WM_ENDPOINT=http://localhost:30000/predict
export SCIWORLD_ENV_URL=http://localhost:30003
export PYTHONPATH=/workspace/user/sciworld_training:$PYTHONPATH

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export WORLD_SIZE=1
export NODE_RANK=0
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA

# Will be updated after SFT completes
MODEL_PATH=${MODEL_PATH:-output/sciworld_mistral_sft/v4-20260429-010313/checkpoint-10}
RUN_NAME="sciworld_mistral_sft_grpo_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --template llama \
    --external_plugins sciworld_plugin.py \
    --reward_funcs sciworld_reward \
    --multi_turn_scheduler sciworld_scheduler \
    --max_turns 50 \
    --completion_length_limit_scope per_round \
    --stop_words '</s>' \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --torch_dtype bfloat16 \
    --dataset /workspace/user/sciworld_training/data/sciworld_rl_split_mistral.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 256 \
    --max_length 16384 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --max_grad_norm 0.5 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant \
    --gradient_checkpointing true \
    --save_steps 20 \
    --save_total_limit 5 \
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
    --output_dir output/sciworld_mistral_sft_grpo
