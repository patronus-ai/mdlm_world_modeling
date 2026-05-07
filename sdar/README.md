# SDAR SFT Training & Evaluation

A minimal harness for full fine-tuning of [SDAR](https://github.com/JetBrains-Research/SDAR)
block-diffusion models via the bundled LlamaFactory fork, plus generation +
scoring utilities.

## Files

| File | Purpose |
|---|---|
| `setup.sh`              | Create venv, clone SDAR, install LlamaFactory fork (editable) + extras. |
| `train_sft.sh`          | `torchrun` wrapper around `SDAR/training/llama_factory_sdar/src/llamafactory/launcher.py`. |
| `configs/sft_example.yaml` | LlamaFactory full-finetune SFT template. |
| `prepare_data.py`       | Normalize chat JSONL into the message-array form LlamaFactory expects. |
| `eval_generate.py`      | Multi-GPU batched block-diffusion generation over eval JSONL files. Uses `SDAR/generate.py`. |
| `evaluate.py`           | Score predictions (exact-match, semantic, ROUGE-L) using the same tag-stripping + JSON canonicalization (sort_keys + compact separators) used in `llm_training` / `wedlm_training`, plus `<think>...</think>` block stripping for SDAR CoT. |
| `.env.example`          | Template for `HF_TOKEN` / `WANDB_API_KEY`. Copy to `.env`. |

## Prerequisites

- NVIDIA GPU(s) with a recent CUDA driver
- Python 3.11 (and optionally [`uv`](https://github.com/astral-sh/uv))
- `git` + network access to clone https://github.com/JetBrains-Research/SDAR
- Disk space for the assembled model directory and training outputs

## Model preparation (manual step)

SDAR models require custom modeling code that is **not** redistributed with
the official HuggingFace weights. Each model directory you want to train
against must combine the two:

1. After `setup.sh`, the SDAR repo lives at `./SDAR`. Inside it,
   `SDAR/training/model/<NAME>/` contains the custom `modeling_*.py` and
   `config.json` for that variant (e.g. `SDAR-4B-Chat`, `SDAR-8B-Chat`).
2. Download the official `*.safetensors` weights from the matching HF repo
   into the **same** directory.
3. Point your config (`model_name_or_path:` in `configs/sft_example.yaml`)
   at that directory.

After step 2, the directory should look like:

```
./model/SDAR-8B-Chat/
├── config.json          # custom (from SDAR/training/model/...)
├── modeling_sdar.py     # custom (from SDAR/training/model/...)
├── model-*.safetensors  # official weights from HuggingFace
└── ...                  # tokenizer files etc.
```

## Quick start

```bash
# 1. Install (creates ./.venv, clones SDAR into ./SDAR, installs LlamaFactory)
bash setup.sh

# 2. Assemble your model directory — see "Model preparation" above.

# 3. Configure secrets
cp .env.example .env
# edit .env: HF_TOKEN (gated weights) and WANDB_API_KEY (only if you want wandb)

# 4. Prepare data — emits a JSONL of message arrays.
python prepare_data.py --input data/raw.jsonl --output data/train.jsonl

# 5. Register the dataset with LlamaFactory.
#    Edit SDAR/training/llama_factory_sdar/data/dataset_info.json and add an
#    entry pointing at ./data/train.jsonl. See LlamaFactory docs for the
#    schema. The key you choose becomes the `dataset:` value in your config.

# 6. Edit configs/sft_example.yaml — at minimum set:
#       model_name_or_path:  ./model/SDAR-8B-Chat       (or your dir)
#       dataset:             <your registry key>
#       template:            qwen3                       (for SDAR-8B-Chat)
#       output_dir:          ./outputs/my-run

# 7. Train
NUM_GPUS=8 CONFIG=configs/sft_example.yaml bash train_sft.sh

# 8. Generate predictions
python eval_generate.py \
    --sdar-dir ./SDAR \
    --model-path ./outputs/my-run/checkpoint-1000 \
    --data-dir ./data \
    --datasets test \
    --output-dir ./outputs/eval_results \
    --num-gpus 8

# 9. Score the predictions
python evaluate.py \
    --input-dir ./outputs/eval_results \
    --datasets test \
    --output-dir ./outputs/eval_results
# For api_bank-style refs (wrapper {api_name, input, output, exception}),
# add: --extract-output-field api_bank
```

## Data format

**Training.** LlamaFactory consumes JSONL where each line is a JSON **array**
of message dicts:

```json
[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

`prepare_data.py` accepts either that form or the `{"messages": [...]}`
wrapper form and emits the array form.

**Eval** (`eval_generate.py --data-dir`). JSONL where each line is an object
with a `messages` field; the **last** message's content is treated as the
ground truth and stripped from the prompt before generation:

```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

## Configuring `train_sft.sh`

| Var | Default | |
|---|---|---|
| `CONFIG`        | `./configs/sft_example.yaml` | LlamaFactory YAML config. |
| `SDAR_DIR`      | `./SDAR`                     | SDAR checkout. |
| `VENV_DIR`      | `./.venv`                    | Python venv. |
| `NUM_GPUS`      | `8`                          | GPUs per node. |
| `MASTER_PORT`   | `12345`                      | torchrun master port. |
| `HF_TOKEN`      | (from `.env`)                | Only if weights are gated. |
| `WANDB_API_KEY` | (from `.env`)                | Only if `report_to: wandb`. |

Key SDAR-specific config fields (already set in `sft_example.yaml`):

- `block_length: 4` — must match the model's architectural block size.
- `neat_packing: true` — required by SDAR's FlexAttention path.
- `truncate_mode: drop` — preserves fixed input shapes for FlexAttention.
- `template: qwen3` — chat template for SDAR-8B-Chat (varies by variant).

## Configuring `eval_generate.py`

Defaults match the values used for SDAR-8B world-modeling evals:

| Flag | Default |
|---|---|
| `--gen-length` | `1024` |
| `--block-length` | `4` |
| `--denoising-steps` | `4` |
| `--per-batch` | `8` (samples per forward pass per GPU) |
| `--prompt-truncate` | `8000` |
| `--temperature` | `1.0` |
| `--confidence-threshold` | `0.85` |
| `--remasking-strategy` | `low_confidence_dynamic` |

Sharding is round-robin across `--num-gpus`; each shard sorts prompts by
length before batching, so padding overhead is small.

## Eval-side normalization

`evaluate.py` applies these transforms to **both** prediction and reference
before comparison (a strict superset of the `llm_training` / `wedlm_training`
behavior):

1. Strip entire `<think>...</think>` blocks (SDAR's CoT is wrapped here).
2. If the result is wrapped in `<tool_response>...</tool_response>`, return
   the inner content; otherwise drop residual `<tool_response>` / `<think>`
   tag tokens.
3. `.strip()` outer whitespace.
4. If the result parses as JSON, re-serialize with `sort_keys=True` and
   compact separators (`",":"`), so `{"b":2,"a":1}` and `{"a": 1, "b": 2}`
   compare equal.

Pass `--extract-output-field <name>` for datasets where the reference is a
wrapper but the model only emits a subset (e.g. api_bank).

By default rows whose prediction starts with `ERROR:` (shard-level inference
failures) are dropped, as are rows whose reference normalizes to empty, and
duplicate rows are de-duplicated by `index` (keep first). Override with
`--keep-errors` / `--keep-empty-ref` / `--no-dedupe`.

## Security

Tokens belong in `.env`, never in YAML configs or shell scripts. `.env`
should be gitignored.
