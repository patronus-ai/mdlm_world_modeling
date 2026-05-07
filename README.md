# Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL

Code for the paper: *Masked Diffusion Language Models are Strong and Steerable Text-Based World Models for Agentic RL*.

## Overview

We formalize text-based world modeling as a **steerable transition-dynamics problem** and show that Masked Diffusion Language Models (MDLMs) — by virtue of bidirectional anchor-aware denoising — produce more coherent, grounded, and diverse environment rollouts than autoregressive LLMs more than 4× their total parameter size. We curate a dataset of **239,403 grounded state–action trajectories** spanning nine open-source environments and twelve frontier model families, fine-tune MDLMs as world models, and demonstrate that GRPO training on MDLM-generated rollouts yields absolute task-success improvements of up to **47%** across LiquidAI, Qwen3, and Mistral agent backbones — without any environment-specific fine-tuning.

### Two-stage pipeline

```
Stage 1 — World Model Training          Stage 2 — Agent RL Training
─────────────────────────────────────   ──────────────────────────────────────────
Fine-tune an MDLM (or AR LLM baseline)  GRPO-train a small agent using the world
on 239,403 trajectories to predict      model as a fast environment simulator.
next environment states.                Evaluated on three held-out environments.

  sdar/          SDAR-8B / 30B-A3B        alfworld/   ALFWorld (text game)
  wedlm/         WeDLM-8B                 sciworld/   ScienceWorld (science tasks)
  llm_training/  AR LLM baselines         appworld/   AppWorld (multi-app APIs)
```

## Repository Structure

```
mdlm_world_modeling/
│
├── sdar/                   SDAR block-diffusion SFT training harness
│   ├── setup.sh            Clone JetBrains-Research/SDAR + LlamaFactory fork
│   ├── train_sft.sh        torchrun wrapper for LlamaFactory SFT
│   ├── prepare_data.py     Convert chat JSONL → LlamaFactory message-array format
│   ├── eval_generate.py    Multi-GPU batched block-diffusion inference
│   ├── evaluate.py         Score predictions (EM / ROUGE-L / semantic match)
│   └── configs/
│       └── sft_example.yaml
│
├── wedlm/                  WeDLM (Tencent) SFT training harness
│   ├── setup.sh            Clone Tencent/WeDLM + install finetune extras
│   ├── train.sh            accelerate launch wrapper
│   ├── prepare_data.py     Convert chat JSONL → WeDLM message-array format
│   ├── eval_generate.py    WeDLM inference engine over eval files
│   ├── evaluate.py         Score predictions (EM / ROUGE-L / semantic match)
│   └── configs/
│       └── example.yaml
│
├── llm_training/           AR LLM baseline training (ms-swift full fine-tuning SFT)
│   ├── setup.sh            Clone ms-swift + flash-attn
│   ├── train.sh            Env-var-driven full fine-tuning SFT launcher
│   ├── eval.sh             Loss-based eval via ms-swift
│   ├── prepare_data.py     Download HF dataset → train/test JSONL
│   └── evaluate.py         Generation-based eval against OpenAI-compatible endpoint
│
├── alfworld/               ALFWorld agent RL (SFT + GRPO)
│   ├── alfworld_plugin.py  ms-swift GRPO scheduler + reward function
│   ├── alfworld_wm_prompt.py  WM prompt builder + deterministic local responder
│   ├── alfworld_prompt.py  Agent system prompt
│   ├── build_dataset.py    Bake per-task WM system prompt (scene walk)
│   ├── build_sft_expert.py Expert demo replay → SFT trajectories
│   ├── eval_parallel.py    Parallel batched eval against real AlfredTWEnv
│   ├── wm_proxy.py         :30000 → lmdeploy :30001
│   ├── alfworld_env_server.py  Real env HTTP wrapper (eval only)
│   ├── run_lfm25_{sft,grpo}.sh      LFM2.5-1.2B scripts (1 GPU)
│   ├── run_qwen3_{sft,grpo}.sh      Qwen3-4B scripts (2 GPUs)
│   ├── run_mistral_sft.sh           Mistral-7B SFT (1 GPU)
│   ├── run_mistral_grpo_2gpus.sh    Mistral-7B GRPO (4 GPUs, zero2)
│   └── GETTING_STARTED.md  End-to-end hardware + reproduction guide
│
├── sciworld/               ScienceWorld agent RL (SFT + GRPO)
│   ├── sciworld_plugin.py  ms-swift GRPO scheduler + reward function
│   ├── sciworld_wm_prompt.py  WM prompt builder + local responder
│   ├── sciworld_env_server.py  Real ScienceWorld HTTP wrapper (eval only)
│   ├── eval_sciworld.py    Eval against real ScienceWorld env
│   ├── create_sft_gpt_agent.py  GPT-5.5 SFT demo generation
│   ├── run_sciworld_lfm_{sft,grpo}.sh    LFM2.5-1.2B scripts
│   ├── run_sciworld_mistral_{sft,grpo}.sh  Mistral-7B scripts
│   └── run_sciworld_qwen3_{sft,grpo_qwenwm}.sh  Qwen3-4B scripts
│
└── appworld/               AppWorld agent RL (SFT + GRPO)
    ├── appworld_plugin.py  ms-swift GRPO scheduler + reward function
    ├── appworld_wm_prompt.py  WM prompt builder + local responder
    ├── appworld_prompt.py  Agent system prompt
    ├── fix_tool_names_and_schemas.py  Tool definitions for local validation
    ├── create_sft_gpt_agent.py  GPT-5.5 SFT demo generation
    ├── wm_proxy.py         :30000 → lmdeploy :30001
    ├── run_{qwen,lfm25}_{sft,grpo}.sh  Training scripts
    └── run_lfm25_sft_grpo_v2.sh        LFM2.5 combined pipeline
```

## Stage 1 — World Model Training

All three harnesses share the same data format (chat JSONL with `{"messages": [...]}` rows) and the same evaluation protocol (exact-match, JSON-semantic match with key-order normalization, ROUGE-L).

### SDAR (best performing — recommended for performance)

SDAR-8B achieves **MAUVE 0.982** on the in-domain split, surpassing all AR baselines including Qwen3.5-35B-A3B (0.932).

```bash
cd sdar/
bash setup.sh                               # clones JetBrains-Research/SDAR

# Assemble model dir: copy modeling_*.py + config.json from SDAR/training/model/SDAR-8B-Chat/
# into ./model/SDAR-8B-Chat/, then add safetensors weights from HuggingFace.

cp .env.example .env                        # set HF_TOKEN, WANDB_API_KEY
python prepare_data.py --input data/train.jsonl --output data/train_lf.jsonl
# Register dataset in SDAR/training/llama_factory_sdar/data/dataset_info.json
# Edit configs/sft_example.yaml (model_name_or_path, dataset, output_dir)

NUM_GPUS=8 CONFIG=configs/sft_example.yaml bash train_sft.sh

# Eval
python eval_generate.py --sdar-dir ./SDAR --model-path ./outputs/checkpoint-N \
    --data-dir ./data --datasets eval --output-dir ./outputs/eval_results --num-gpus 8
python evaluate.py --input-dir ./outputs/eval_results --datasets eval \
    --output-dir ./outputs/eval_results
```

Key SDAR-specific config fields (already set in `sft_example.yaml`): `block_length: 4`, `neat_packing: true`, `truncate_mode: drop`, `template: qwen3`.

### WeDLM (best performing for speed)

```bash
cd wedlm/
bash setup.sh                               # clones Tencent/WeDLM
# (Optional) Install MagiAttention for faster long-sequence training.
# Otherwise set attention_backend: dense in configs/example.yaml.

cp .env.example .env
python prepare_data.py --input data/train.jsonl --output data/train_we.jsonl
# Edit configs/example.yaml (model_path, train_data, output_dir, attention_backend)

NUM_GPUS=8 CONFIG=configs/example.yaml bash train.sh

python eval_generate.py --model-path ./outputs/my-run/final \
    --data-dir ./data --datasets eval --output-dir ./outputs/eval_results
python evaluate.py --input-dir ./outputs/eval_results --datasets eval \
    --output-dir ./outputs/eval_results
```

### AR LLM Baselines (ms-swift)

Used to train Qwen3-8B, GPT-OSS-20B, and other AR baselines.

```bash
cd llm_training/
bash setup.sh                               # clones ms-swift; pass FLASH_ATTN_WHL= for flash-attn
cp .env.example .env

MODEL=Qwen/Qwen3-8B \
TRAIN_FILE=./data/train.jsonl \
VAL_FILE=./data/test.jsonl \
RUN_NAME=qwen3_8b_wm \
  bash train.sh

# Eval via loss (no generation):
MODEL_DIR=./output/qwen3_8b_wm EVAL_FILES="./data/test.jsonl" bash eval.sh

# Eval via generation (requires a running vLLM server):
python evaluate.py --api-url http://localhost:8000/v1/chat/completions \
    --model qwen3_8b_wm --data-dir ./data --datasets test \
    --output-dir ./output/eval_results
```

Key training knobs are controlled through environment variables; see `llm_training/README.md` for the full table.

## Stage 2 — Agent RL Training

All three environments use the same architecture:

```
Agent (ms-swift GRPO)  ──action──>  Plugin
                                      ├── local responder  (deterministic, ~85% of actions)
                                      └── WM proxy :30000  ──>  lmdeploy :30001  (SDAR WM)
                                                                      ↑
                                                             SDAR_world_model
```

The real environment is used **only for evaluation**, never during training. Reward is computed from trajectory structure (no live env access).

### Starting the World Model stack

```bash
# SDAR-WM (recommended — 2× 80GB GPUs)
SDAR_PATH=/path/to/SDAR_world_model
CUDA_VISIBLE_DEVICES=2,3 lmdeploy serve api_server $SDAR_PATH \
    --tp 2 --server-port 30001 --session-len 65536 \
    --max-batch-size 8 --dllm-block-length 8 --dllm-denoising-steps 1 &

# OR: Qwen3.5-35B-A3B WM (4× 80GB GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen3.5-35B-A3B_world_model --port 30001 \
    --tensor-parallel-size 4 --language-model-only --dtype bfloat16 &

# WM proxy (runs on agent host, translates payload format)
python wm_proxy.py &    # :30000 → :30001
```

### ALFWorld

Three agent backbones. All scripts expect the WM stack on `:30000`.

```bash
cd alfworld/
export ALFWORLD_DATA=$PWD/data
alfworld-download

# Build training data
python build_sft_expert.py --workers 16 --limit 1500 --out data/alfworld_sft.jsonl
python build_dataset.py --split train --shuffle --workers 16 --limit 300 \
    --out data/alfworld_rl.jsonl
python build_dataset.py --split eval_in_distribution --out data/alfworld_eval_id.jsonl

# LFM2.5-1.2B (1 GPU)
CUDA_VISIBLE_DEVICES=0 bash run_lfm25_alfworld_sft.sh
MODEL_PATH=output/alfworld_lfm25_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0 bash run_lfm25_alfworld_grpo.sh

# Qwen3-4B (2 GPUs, zero2)
CUDA_VISIBLE_DEVICES=0,1 bash run_qwen3_alfworld_sft.sh
MODEL_PATH=output/alfworld_qwen3_4b_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0,1 bash run_qwen3_alfworld_grpo.sh

# Mistral-7B (4 GPUs, zero2)
CUDA_VISIBLE_DEVICES=0 bash run_mistral_sft.sh
MODEL_PATH=output/alfworld_mistral_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_mistral_grpo_2gpus.sh

# Eval (real env — start alfworld_env_server.py first)
python eval_parallel.py $CKPT data/alfworld_eval_id.jsonl /tmp/out.jsonl 0 \
    --max-turns 50 --parallel 32
```

See `alfworld/GETTING_STARTED.md` for the full hardware guide (WM stack, data build, common pitfalls).

### ScienceWorld

```bash
cd sciworld/
# Start real env server (eval only): python sciworld_env_server.py &

# LFM2.5-1.2B
CUDA_VISIBLE_DEVICES=0 bash run_sciworld_lfm_sft.sh
MODEL_PATH=output/sciworld_lfm25_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0 bash run_sciworld_lfm_grpo.sh          # SDAR WM

# Mistral-7B (SDAR WM or Qwen WM)
CUDA_VISIBLE_DEVICES=0 bash run_sciworld_mistral_sft.sh
CUDA_VISIBLE_DEVICES=0 bash run_sciworld_mistral_grpo.sh         # SDAR WM
CUDA_VISIBLE_DEVICES=0 bash run_sciworld_mistral_grpo_qwenwm.sh  # Qwen3.5 WM

# Qwen3-4B
CUDA_VISIBLE_DEVICES=0 bash run_sciworld_qwen3_sft.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_sciworld_qwen3_grpo_qwenwm.sh

# Eval
CUDA_VISIBLE_DEVICES=0 python eval_sciworld.py $CKPT \
    data/sciworld_test_split.jsonl <label> 0
```

### AppWorld

```bash
cd appworld/

# SFT demos (requires GPT access)
python create_sft_gpt_agent.py   # → data/appworld_sft_gpt_agent.jsonl

# LFM2.5-1.2B
CUDA_VISIBLE_DEVICES=0 bash run_lfm25_sft.sh
MODEL_PATH=output/appworld_lfm25_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0 bash run_lfm25_grpo.sh

# Qwen3-4B
CUDA_VISIBLE_DEVICES=0 bash run_qwen_sft.sh
MODEL_PATH=output/appworld_qwen3_sft/.../checkpoint-N \
  CUDA_VISIBLE_DEVICES=0 bash run_qwen_grpo.sh
```

## Prerequisites

- NVIDIA GPU(s) with CUDA 12.4+ driver
- Python 3.11 (3.12 for `llm_training/`)
- [`uv`](https://github.com/astral-sh/uv) (recommended)
- Each subdirectory's `setup.sh` handles all Python dependencies

## Environment Variables

Each component reads credentials from a local `.env` file (never hardcoded). Copy `.env.example` → `.env` and fill in:

| Variable | Required for |
|---|---|
| `HF_TOKEN` | Downloading gated model weights |
| `WANDB_API_KEY` | Training run logging |
| `OPENAI_API_KEY` | GPT-based SFT demo generation; OpenAI eval baselines |

## Notes

- The training dataset (239,403 trajectories split into `train.jsonl`, `eval.jsonl`, `test.jsonl`, `ood_test.jsonl`) is released separately on HuggingFace.
- All GRPO experiments use [ms-swift](https://github.com/modelscope/ms-swift) (commit `43b5d8e`) with the `swift rlhf --rlhf_type grpo` entrypoint.
- World model inference uses [lmdeploy](https://github.com/InternLM/lmdeploy) for SDAR/block-diffusion models and [vLLM](https://github.com/vllm-project/vllm) for AR models.
