# LLM Training & Evaluation (ms-swift Full Fine-Tuning SFT)

A minimal, generalized harness for full supervised fine-tuning of chat LLMs
with [ms-swift](https://github.com/modelscope/ms-swift), plus evaluation
helpers.

## Files

| File | Purpose |
|---|---|
| `setup.sh`         | Create venv, clone ms-swift, install deps + flash-attn. |
| `train.sh`         | Generalized full fine-tuning SFT launcher (env-var driven). |
| `eval.sh`          | Loss-based eval over JSONL files via `ms-swift sft --do_train false`. |
| `evaluate.py`      | Generation-based eval against an OpenAI-compatible endpoint (e.g. vLLM). Reports exact-match, JSON-semantic match, ROUGE-L. |
| `prepare_data.py`  | Download a HuggingFace chat dataset and emit `train.jsonl` / `test.jsonl`. |
| `.env.example`     | Template for tokens (HF_TOKEN, WANDB_API_KEY). Copy to `.env`. |

## Prerequisites

- NVIDIA GPU(s) with a recent CUDA driver
- Python 3.12 and [`uv`](https://github.com/astral-sh/uv)
- For multi-GPU training: any reasonably modern CUDA + NCCL setup
- Disk space for the model checkpoint(s) and ms-swift checkout

## Quick start

```bash
# 1. Install
bash setup.sh
# Optionally pass a flash-attn wheel matching your CUDA / torch / Python combo:
#   FLASH_ATTN_WHL=https://.../flash_attn-...whl bash setup.sh

# 2. Configure secrets
cp .env.example .env
# edit .env and set HF_TOKEN and (optionally) WANDB_API_KEY

# 3. Prepare data — emits ./data/train.jsonl and ./data/test.jsonl
HF_TOKEN=$HF_TOKEN python prepare_data.py \
    --repo my-org/my-dataset \
    --train-configs default \
    --test-configs default

# 4. Train
MODEL=Qwen/Qwen3-8B \
TRAIN_FILE=./data/train.jsonl \
VAL_FILE=./data/test.jsonl \
RUN_NAME=qwen3_demo \
  bash train.sh

# 5. Loss-based eval on held-out files
MODEL_DIR=./output/qwen3_demo \
EVAL_FILES="./data/test.jsonl" \
  bash eval.sh

# 6. Generation-based eval (start a vLLM server first, then:)
python evaluate.py \
    --api-url http://localhost:8000/v1/chat/completions \
    --model qwen3_demo \
    --data-dir ./data \
    --datasets test \
    --output-dir ./output/eval_results
```

## Data format

Both `train.sh` and `evaluate.py` expect JSONL where each line is:

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

For `evaluate.py`, the **last** assistant message is treated as the
ground-truth target and stripped from the prompt sent to the endpoint.

## Configuring `train.sh`

`train.sh` reads everything from environment variables (and `.env` if
present). The most common knobs:

| Var | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-8B` | Model id (HF or ModelScope). |
| `USE_HF` | `true` | Set `false` for ModelScope. |
| `TEMPLATE` | _(auto)_ | Override chat template, e.g. `chatml`. |
| `TRAIN_FILE` / `VAL_FILE` | `./data/train.jsonl` / `./data/test.jsonl` | Input JSONL files. |
| `OUTPUT_DIR` | `./output/$RUN_NAME` | Checkpoint dir. |
| `RUN_NAME` | `sft_run` | Wandb run name. |
| `NUM_GPUS` | `8` | GPUs per node. |
| `MASTER_PORT` | `29500` | torch.distributed port. |
| `ATTN_IMPL` | `flash_attn` | `flash_attn` \| `eager` \| `sdpa`. |
| `EPOCHS` | `2` | |
| `PER_DEVICE_TRAIN_BATCH_SIZE` / `GRAD_ACCUM_STEPS` | `2` / `4` | |
| `LR` / `WARMUP_RATIO` / `LR_SCHED` | `2e-5` / `0.03` / `cosine` | |
| `MAX_LEN` | `8192` | |
| `PACKING` | `true` | Sequence packing (set `false` if your model dislikes it). |
| `DEEPSPEED` | `zero2` | `zero2` \| `zero3` \| `none`. |
| `REPORT_TO` | `wandb` | Set `none` to disable. |

For example, to switch to a different base model with eager attention and
no packing:

```bash
MODEL=nvidia/SomeModel TEMPLATE=chatml \
ATTN_IMPL=eager PACKING=false \
RUN_NAME=somemodel_run \
  bash train.sh
```

## Tips

- **Disabling wandb**: `REPORT_TO=none bash train.sh`.
- **Single-GPU debug run**: `NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0
  PER_DEVICE_TRAIN_BATCH_SIZE=1 DEEPSPEED=none bash train.sh`.
- **No flash-attn wheel for your stack**: skip `FLASH_ATTN_WHL` in
  `setup.sh` and run with `ATTN_IMPL=eager`.

## Security note

Tokens belong in `.env`, never in shell scripts. `.env` should be added to
`.gitignore` and never checked in.
