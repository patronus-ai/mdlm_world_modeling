# ScienceWorld Training Pipeline

SFT + GRPO training for ScienceWorld using a World Model for rollout simulation.

## Setup

```bash
# Training venv
python3.11 -m venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm lmdeploy requests python-dotenv openai wandb hf_transfer
uv pip install -e /path/to/ms-swift

# ScienceWorld env (separate venv or same)
uv pip install scienceworld==1.2.3

cat > .env <<EOF
OPENAI_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
EOF
```

## Services

```bash
# 1. Env server (uses scienceworld venv)
python sciworld_env_server.py &   # :30003

# 2. World Model (see AppWorld README for WM setup on :30001)
python wm_proxy.py &              # :30000 → :30001
```

## Data Generation

```bash
# Task splits (requires scienceworld)
python data/create_sciworld_splits.py    # → data/sciworld_rl_split.jsonl, data/sciworld_test_split.jsonl

# SFT demos via GPT-5.5
python create_sft_gpt_agent.py           # → data/sciworld_sft_gpt_perfect.jsonl

# Mistral-formatted data (system prompt merged into first user message)
# See create_sciworld_splits.py — Mistral v0.3 requires this
```

## Train

```bash
# LFM-1.2B
bash run_sciworld_lfm_sft.sh
bash run_sciworld_lfm_grpo.sh

# Mistral-7B (template=llama, data=sciworld_rl_split_mistral.jsonl)
bash run_sciworld_mistral_sft.sh
bash run_sciworld_mistral_grpo.sh            # SDAR WM
bash run_sciworld_mistral_grpo_qwenwm.sh     # Qwen3.5 WM

# Qwen3-4B
bash run_sciworld_qwen3_sft.sh
bash run_sciworld_qwen3_grpo_qwenwm.sh
```

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sciworld.py <checkpoint> data/sciworld_test_split.jsonl <label> 0
# Use sciworld_test_split_mistral.jsonl for Mistral
```

## Key Design

- Local responder handles teleport, wait, inventory, focus validation deterministically
- WM called only for look around, pick up, open, and other state-changing actions
- Reward replays actions against real ScienceWorld env; clamped to [-0.2, 1.0]
- No-submission penalty (-0.15) prevents explore-forever behavior
