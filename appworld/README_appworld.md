# AppWorld Training Pipeline

SFT + GRPO training for AppWorld using a World Model for rollout simulation.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install vllm lmdeploy requests python-dotenv openai wandb hf_transfer
uv pip install -e /path/to/ms-swift   # git clone https://github.com/modelscope/ms-swift (commit 43b5d8e)

# AppWorld env
uv pip install appworld

# Create .env
cat > .env <<EOF
OPENAI_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
EOF
```

## World Model

```bash
# Option 1: SDAR (diffusion LM) via lmdeploy
SDAR=$(ls -d ~/.cache/huggingface/hub/models--ANONYMOUS--SDAR_world_model_v2/snapshots/*/)
CUDA_VISIBLE_DEVICES=6,7 lmdeploy serve api_server $SDAR --tp 2 --server-port 30001

# Option 2: Qwen3.5-35B-A3B via vLLM
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model ANONYMOUS/Qwen3.5-35BA3B_world_model_v2 --port 30001 \
    --tensor-parallel-size 4 --language-model-only --dtype bfloat16

# WM proxy (translates payload format)
python wm_proxy.py &   # listens :30000, forwards to :30001
```

## Train

```bash
# 1. SFT
bash run_qwen_sft.sh

# 2. GRPO (set MODEL_PATH to SFT checkpoint)
MODEL_PATH=output/appworld_qwen3_sft/checkpoint-32 bash run_qwen_grpo.sh
```

## Files

| File | Purpose |
|------|---------|
| `appworld_plugin.py` | GRPO scheduler + reward function |
| `appworld_wm_prompt.py` | WM prompt builder + deterministic local responder |
| `appworld_prompt.py` | Agent system prompt |
| `fix_tool_names_and_schemas.py` | Tool definitions for local validation |
| `wm_proxy.py` | HTTP proxy: :30000 → :30001 |
| `create_sft_gpt_agent.py` | Generate SFT demos with GPT |
| `data/appworld_rl_split.jsonl` | RL training data |
| `data/appworld_sft_gpt_agent.jsonl` | SFT demonstrations |
