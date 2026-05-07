# ALFWorld Training Pipeline

SFT + GRPO training for ALFWorld using a World Model for rollout simulation.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install alfworld "tatsu==5.8.3"    # tatsu pin avoids textworld breakage
uv pip install vllm lmdeploy requests python-dotenv openai wandb hf_transfer
uv pip install -e /path/to/ms-swift

export ALFWORLD_DATA=$PWD/data
alfworld-download

cat > .env <<EOF
OPENAI_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
EOF
```

Note: ALFWorld's import chain requires `sys.setrecursionlimit(50000)` — all scripts include this.

## Services

```bash
# 1. Env server
python alfworld_env_server.py &     # :30003

# 2. World Model (see AppWorld README for WM setup on :30001)
python wm_proxy.py &                # :30000 → :30001 (includes repetition_penalty=1.3)
```

## Data Generation

```bash
# Eval set (140 in-distribution games)
python -c "import sys; sys.setrecursionlimit(50000); ..." # see build_dataset.py

# RL training set (300 games)
# see build_dataset.py --split train --limit 300 --shuffle

# SFT expert demos (parallelized, ~1000 winning trajectories)
# see build_sft_expert.py

# Mistral-formatted data (system prompt merged into first user message)
```

## Train

```bash
# Mistral-7B (4 GPUs, zero2)
bash run_mistral_sft.sh
bash run_mistral_grpo_2gpus.sh

# LFM-1.2B (1 GPU)
bash run_lfm25_alfworld_sft.sh
bash run_lfm25_alfworld_grpo.sh

# Qwen3-4B (2 GPUs, zero2)
bash run_qwen3_alfworld_sft.sh
bash run_qwen3_alfworld_grpo.sh
```

## Eval

```bash
# Fast eval — loads model once, batched generation
python eval_fast.py <checkpoint> --dataset data/alfworld_eval_id.jsonl --gpus 0 --max-turns 35

# Parallel eval (original)
python eval_parallel.py <checkpoint> data/alfworld_eval_id.jsonl <out.jsonl> <gpu_id>
```

## Key Design

- Local deterministic responder handles ~85% of actions (go to, take, move, open, close, clean, heat, cool, use)
- WM called only for first-visit receptacle contents, examine, and look
- Receptacle state (open/closed/contents) tracked across turns from baked initial scene walk
- Instance number resolution: "take pot" → "take pot 1" via fuzzy matching
- Reward: win=1.0, take+place=0.25, take+transform+place=0.35, take-only=0.03, explore-only=-0.05
- ABAB loop detection + low-diversity kill (≤3 unique actions in 20 turns)
