#!/bin/bash
cd /workspace/user/sciworld_training
source /workspace/user/trl_training/.venv/bin/activate
export $(grep -v '^#' /workspace/user/trl_training/.env | sed 's/ =/=/g' | xargs)
export HF_HOME=/workspace/.cache/huggingface/
export USE_HF=1
export WM_ENDPOINT=http://localhost:30000
export SCIWORLD_ENV_URL=http://localhost:30004
export PYTHONPATH=/workspace/user/sciworld_training:$PYTHONPATH
export TMPDIR=/workspace/tmp

export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((RANDOM % 10000 + 40000))
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
unset NUM_NODES HOST_NODE_ADDR NODE_ROLE NODE_ADDR NCCL_IB_HCA WORLD_SIZE

MODEL_PATH=output/sciworld_qwen3_sft/v0-20260502-002645/checkpoint-10
RUN_NAME="sciworld_qwen3_grpo_qwenwm_$(date +%Y%m%d_%H%M%S)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT="wm_bench" \
WANDB_RUN_NAME="$RUN_NAME" \
swift rlhf \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --template qwen3 \
    --external_plugins sciworld_plugin.py \
    --reward_funcs sciworld_reward \
    --multi_turn_scheduler sciworld_scheduler \
    --max_turns 50 \
    --completion_length_limit_scope per_round \
    --stop_words '<|im_end|>' '<|endoftext|>' \
    --tuner_type full \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_max_model_len 16384 \
    --sleep_level 1 \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --dataset data/sciworld_rl_split.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0 \
    --max_completion_length 256 \
    --max_length 16384 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
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
    --output_dir output/sciworld_qwen3_grpo_qwenwm
