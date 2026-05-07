# WeDLM SFT Training & Evaluation

A minimal harness for SFT fine-tuning of [WeDLM](https://github.com/Tencent/WeDLM)
diffusion language models, plus generation + scoring utilities.

## Files

| File | Purpose |
|---|---|
| `setup.sh`            | Create venv, clone WeDLM, install finetune extras. **Does not install MagiAttention.** |
| `train.sh`            | Wrapper around `accelerate launch ... WeDLM/finetune/train.py --config <YAML>`. |
| `configs/example.yaml`| Editable config template (model, data, attention backend, LR, etc.). |
| `prepare_data.py`     | Convert `{"messages": [...]}` JSONL into the message-array JSONL the trainer expects. |
| `eval_generate.py`    | Run the WeDLM inference engine over eval JSONL files and write predictions. |
| `evaluate.py`         | Score those predictions (exact-match, semantic-match, ROUGE-L) using the same tag-stripping + JSON-canonicalization (sort_keys + compact separators) used in `llm_training`. |
| `.env.example`        | Template for `HF_TOKEN` / `WANDB_API_KEY`. Copy to `.env`. |

## Prerequisites

- NVIDIA GPU(s) with a recent CUDA driver
- Python 3.11 (and optionally [`uv`](https://github.com/astral-sh/uv))
- Network access to clone https://github.com/Tencent/WeDLM
- Disk space for the base model checkpoint and training outputs

## MagiAttention (manual step)

`configs/example.yaml` lets you choose `attention_backend: dense` or `magi`.

- **`dense`**: PyTorch SDPA with a 2D mask. No extra installation required.
- **`magi`**: [MagiAttention](https://github.com/SandAI-org/MagiAttention) flex
  flash attention. Faster on long sequences, but **you must install it
  yourself** — it involves CUDA kernel compilation and is non-trivial.

If you want `magi`, follow the upstream guide
(https://github.com/SandAI-org/MagiAttention) **before running training**, and
make sure your CUDA toolkit version matches your PyTorch CUDA version.
Otherwise, leave `attention_backend: dense` and skip MagiAttention entirely.

## Quick start

```bash
# 1. Install (creates ./.venv, clones WeDLM into ./WeDLM, installs extras)
bash setup.sh

# 2. (Optional) Install MagiAttention yourself — see section above.

# 3. Configure secrets
cp .env.example .env
# edit .env to set HF_TOKEN (gated models) and/or WANDB_API_KEY

# 4. Prepare data
#    Input is a JSONL of {"messages": [...]} rows; output is a JSONL of
#    raw [..., {role, content}, ...] arrays as the trainer expects.
python prepare_data.py --input data/raw.jsonl --output data/train.jsonl

# 5. Edit configs/example.yaml — at minimum set:
#       model_path:        e.g. tencent/WeDLM-8B-Base, or a local path
#       train_data:        ./data/train.jsonl
#       output_dir:        ./outputs/my-run
#       attention_backend: dense   (or "magi" if installed)

# 6. Train
NUM_GPUS=8 CONFIG=configs/example.yaml bash train.sh

# 7. Generate predictions on eval files
#    Expects eval JSONL files at <data-dir>/eval_<name>.jsonl,
#    each row a {"messages": [...]} where the last message is ground truth.
python eval_generate.py \
    --model-path ./outputs/my-run/final \
    --data-dir ./data \
    --datasets test \
    --output-dir ./outputs/eval_results

# 8. Score the predictions
python evaluate.py \
    --input-dir ./outputs/eval_results \
    --datasets test \
    --output-dir ./outputs/eval_results
# For api_bank-style refs (wrapper {api_name, input, output, exception}),
# add: --extract-output-field api_bank
```

## Data format

**Training** (`train_data:` in the config) — JSONL where each line is a JSON
**array** of message dicts:

```json
[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

**Eval** (`eval_generate.py --data-dir`) — JSONL where each line is an object
with a `messages` field; the **last** assistant message is treated as the
ground truth and stripped from the prompt:

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

`prepare_data.py` converts the eval-style format into the training-style
format if you need it.

## Configuring `train.sh`

| Var | Default | |
|---|---|---|
| `CONFIG`           | `./configs/example.yaml` | YAML config to load. |
| `WEDLM_DIR`        | `./WeDLM`                | WeDLM checkout. |
| `VENV_DIR`         | `./.venv`                | Python venv. |
| `NUM_GPUS`         | `8`                      | Multi-GPU when >1, else single GPU. |
| `MIXED_PRECISION`  | `bf16`                   | `bf16` \| `fp16` \| `no`. |
| `HF_TOKEN`         | (from `.env`)            | Only if model is gated. |
| `WANDB_API_KEY`    | (from `.env`)            | Only if `use_wandb: true`. |

## Eval-side normalization

`evaluate.py` applies these transforms to **both** prediction and reference
before comparison:

1. Strip `<tool_response>...</tool_response>` and `<think>...</think>` tags.
2. `.strip()` outer whitespace.
3. If the result parses as JSON, re-serialize with `sort_keys=True` and
   compact separators (`",":"`), so `{"b":2,"a":1}` and `{"a": 1, "b": 2}`
   compare equal.

For datasets where the reference is a wrapper but the model only emits a
subset (e.g. `api_bank` references include `{api_name, input, output,
exception}` but the model predicts the response payload), pass
`--extract-output-field <name>` so the reference is reduced to
`{output, exception}` before comparison.

By default, rows flagged `skipped_overlong` (set by `eval_generate.py` when a
prompt exceeds the model context budget) and rows whose reference normalizes
to empty are dropped from metric averages. Override with `--keep-overlong` /
`--keep-empty-ref`.

## Security

Tokens belong in `.env`, never in YAML configs or shell scripts. `.env`
should be gitignored.
