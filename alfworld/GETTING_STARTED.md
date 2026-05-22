# ALFWorld Agent Training — Getting Started

Train tool-using agents on ALFWorld textworld tasks using SFT + GRPO. The
training loop talks to the **SDAR World Model** (`ANONYMOUS/SDAR_world_model_v2`)
via HTTP, while a deterministic local responder handles cases the WM doesn't
need to predict. The real `AlfredTWEnv` is reserved for evaluation only.

**Note**: The local responder is **completely optional** and can be disabled by setting `ALFWORLD_WM_GUARD=0`. When disabled, all actions are sent to the World Model instead of being handled locally. See the "Local Responder (Optional)" section below for details.

## Architecture

```
TRAINING                                    EVALUATION
  agent  --action-->  alfworld_plugin         agent  --action-->  alfworld_env_server
                       |                                            |
                       v                                            v
                  local responder                              real AlfredTWEnv
                       | (defer)
                       v
                   wm_proxy:30000  -->  lmdeploy:30001  -->  SDAR WM
```

Reward is computed by `check_goal_satisfied()` from the trajectory itself —
no real-env access during training.

## Local Responder (Optional)

The local responder is **completely optional**. It provides deterministic responses for ~85% of actions (navigation, inventory, object manipulation) without calling the World Model, which speeds up training and reduces hallucination.

**To disable the local responder**:
```bash
export ALFWORLD_WM_GUARD=0
```

Or add it to your `.env` file or inline in training scripts:
```bash
ALFWORLD_WM_GUARD=0 bash run_lfm25_grpo.sh
```

When disabled (`ALFWORLD_WM_GUARD=0`), all actions are sent to the World Model for responses. The local responder is enabled by default (`ALFWORLD_WM_GUARD=1`).

## 1. Hardware & disk

- **Training**: 1×80GB GPU for the agent (LFM2.5-1.2B or Qwen3-4B), 2×80GB GPUs for the WM (tp=2)
- **Eval**: 1×80GB GPU
- **Disk**: ~50GB for HF model cache + ~350MB ALFWorld data + checkpoints

## 2. Environment setup

```bash
# 0. clone or copy this directory to your cluster, then cd into it
cd alfworld

# 1. uv venv (Python 3.11)
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2. core deps — pinned set that's known to coexist
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "alfworld" "tatsu==5.8.3"        # tatsu pin avoids textworld breakage
uv pip install -e /path/to/ms-swift             # ms-swift checkout, editable
uv pip install vllm lmdeploy json_repair requests wandb openai hf_transfer

# 3. ALFWorld data download (~315MB)
export ALFWORLD_DATA=$PWD/data
alfworld-download

# 4. environment variables (.env file with the keys you need)
cat > .env <<EOF
WANDB_API_KEY=...
HF_TOKEN=...                 # required for SDAR WM (private repo)
OPENAI_API_KEY=...           # only if you want to eval gpt-* baselines
EOF
```

### libcudart.so.13 fix

vllm 0.20+ ships against CUDA 13. The run scripts already export:

```bash
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

If you launch python directly, do the same.

## 3. Pre-download model weights

ms-swift defaults to ModelScope (slow); prefer HuggingFace with hf_transfer:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1 USE_HF=1
hf download LiquidAI/LFM2.5-1.2B-Instruct
hf download Qwen/Qwen3-4B-Instruct-2507
hf download mistralai/Mistral-7B-Instruct-v0.3   # if you want to train Mistral too
hf download ANONYMOUS/SDAR_world_model_v2       # gated; needs HF_TOKEN
```

The SDAR config is missing `pad_token_id` — patch it once after download:

```bash
SDAR=$(ls -d /workspace/.cache/huggingface/hub/models--ANONYMOUS--SDAR_world_model_v2/snapshots/*/ | head -1)
python -c "import json,sys; p='${SDAR}config.json'; d=json.load(open(p)); d['pad_token_id']=151643; json.dump(d, open(p,'w'), indent=2)"
```

## 4. Start the World Model stack (training only)

Two GPUs (e.g. 2,3) for SDAR via lmdeploy + a tiny proxy on the agent's host.

```bash
SDAR=$(ls -d /workspace/.cache/huggingface/hub/models--ANONYMOUS--SDAR_world_model_v2/snapshots/*/ | head -1)
CUDA_VISIBLE_DEVICES=2,3 lmdeploy serve api_server $SDAR \
    --tp 2 --server-port 30001 --session-len 65536 \
    --max-batch-size 8 --dllm-block-length 8 --dllm-denoising-steps 1 &

python wm_proxy.py &      # listens on :30000, forwards to :30001
```

The proxy auto-resolves the model id from lmdeploy's `/v1/models`, so no hardcoded paths.

## 5. Start the real-env server (eval only)

```bash
python alfworld_env_server.py &   # listens on :30003
```

## 6. Build datasets

```bash
export ALFWORLD_DATA=$PWD/data

# (a) SFT data — replays handcoded expert plans, parallelised
python build_sft_expert.py --workers 16 --limit 1500 --out data/alfworld_sft.jsonl

# (b) RL training data — bakes per-task wm_system_prompt with full receptacle map
python build_dataset.py --split train --shuffle --workers 16 --limit 300 --out data/alfworld_rl.jsonl

# (c) Eval set — full 140-game in-distribution split
python build_dataset.py --split eval_in_distribution --out data/alfworld_eval_id.jsonl
# (d) Optional: out-of-distribution eval split
python build_dataset.py --split eval_out_of_distribution --out data/alfworld_eval_ood.jsonl
```

The "env-walk" inside `build_dataset.py` visits every receptacle once (opening
closed ones) and bakes contents into `wm_system_prompt` so the local responder
can answer `go to X` deterministically without WM hallucination.

## 7. Train

### LFM2.5-1.2B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0 bash run_lfm25_sft.sh
# pick a checkpoint (e.g. checkpoint-300) once SFT finishes
MODEL_PATH=output/alfworld_lfm25_sft/<run>/checkpoint-300 \
  CUDA_VISIBLE_DEVICES=0 bash run_lfm25_grpo.sh
```

### Qwen3-4B-Instruct-2507
```bash
CUDA_VISIBLE_DEVICES=0 bash run_qwen3_sft.sh
MODEL_PATH=output/alfworld_qwen3_4b_sft/<run>/checkpoint-300 \
  CUDA_VISIBLE_DEVICES=0 bash run_qwen3_grpo.sh
```

Things to know about the run scripts:
- LFM2.5 uses `--template chatml` and LR 2e-5 for SFT, 2e-6 for GRPO.
- Qwen3 uses `--template qwen3` and LR 2e-6 for both (higher LR breaks tool calling per prior findings).
- GRPO checkpoints don't always include tokenizer files; if eval errors with
  "Repo id must be...", copy `tokenizer.json` and `tokenizer_config.json` from
  the SFT checkpoint into the GRPO checkpoint dir.

## 8. Evaluate

Use `eval_parallel.py` against the real env server:

```bash
# Pick any checkpoint (SFT, GRPO, or base model path)
CKPT=output/alfworld_lfm25_grpo/<run>/checkpoint-XXX

# All 140 in-distribution games on GPU 4, 32 concurrent rollouts
python eval_parallel.py $CKPT data/alfworld_eval_id.jsonl /tmp/eval_out.jsonl 4 \
    --max-turns 50 --parallel 32
```

`eval_parallel.py` advances all in-flight episodes in lockstep and batches the
generate calls through vLLM. It retries on transient connection errors — make
sure the env server has those retries handled (it does).

For an OpenAI baseline:
```bash
python eval_openai.py gpt-5.5 data/alfworld_eval_id.jsonl /tmp/oai_out.jsonl \
    --max-turns 50 --concurrency 16
```

For step-by-step debugging on the real env (sequential, one episode at a time):
```bash
python baseline_trajectories.py $CKPT data/alfworld_eval_id.jsonl /tmp/dbg.jsonl 4 --max-turns 50
```

## 9. Quick sanity check

```bash
# Without launching anything heavy:
python smoke_env.py        # walks an expert game, verifies env wins
```

## File map

```
alfworld_prompt.py         — Agent system prompt + format_user_turn helper
alfworld_wm_prompt.py      — WM prompt builder + deterministic local responder
                              + admissible_commands generator + goal checker
alfworld_plugin.py         — ms-swift GRPO scheduler + reward function
alfworld_env_server.py     — Real AlfredTWEnv HTTP wrapper (eval only)
wm_proxy.py                — wm_proxy:30000 → lmdeploy:30001
build_dataset.py           — Walk env, bake scene+goal into wm_system_prompt
build_sft_expert.py        — Replay handcoded expert → SFT trajectories
baseline_trajectories.py   — Sequential vLLM eval (debug)
eval_parallel.py           — Parallel batched eval (production)
eval.py                    — Sequential eval, light variant
eval_openai.py             — OpenAI Responses-API eval (gpt-5.5 etc)
smoke_env.py               — 30-second env sanity check
analyze_*.py               — Failure-mode dumpers (optional)
run_lfm25_sft.sh           — SFT script for LFM2.5-1.2B
run_lfm25_grpo.sh          — GRPO script for LFM2.5-1.2B
run_qwen3_sft.sh           — SFT script for Qwen3-4B-Instruct-2507
run_qwen3_grpo.sh          — GRPO script for Qwen3-4B-Instruct-2507
run_qwen_sft.sh            — Earlier Qwen variant scripts (legacy)
run_qwen_grpo.sh
data/                      — ALFWORLD_DATA + your built jsonls
```

## Common pitfalls

- **`max_tokens must be at least 1, got 0`** during GRPO: context overflowed
  the per-round budget. Bump `--max_length` (e.g. 32768) and lower
  `--max_completion_length` if needed.
- **GRPO checkpoints missing tokenizer files**: copy `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja` from the SFT checkpoint.
- **vLLM `libcudart.so.13: cannot open`**: export the cu13 LD_LIBRARY_PATH
  shown in §2 above.
- **Connection-reset under load**: env_server uses a coarse lock + retries
  in `eval_parallel.py`. Don't reduce the retry count.
- **Empty admissible_commands during GRPO** breaks training — make sure the
  scheduler in `alfworld_plugin.py` calls `admissible_commands(state)` on
  every step (it already does).
- **Goal-target mismatch**: agents on long episodes tend to drop the goal text
  from attention (it's only in the initial user turn). If you see the agent
  picking the wrong object class, consider repeating the goal in every user
  turn or adding it to the system prompt at every step.

## Reproducing a full run

```bash
# 1. setup (§2-3)
# 2. start WM stack (§4) and env server (§5)
# 3. build data (§6)
# 4. SFT then GRPO (§7) — pick LFM2.5 or Qwen3
# 5. eval (§8) — eval_parallel against real env
```

End-to-end time: SFT ~30-40 min on a 1.2B model, ~40-60 min on a 4B model;
GRPO ~2-4h depending on `num_train_epochs` and rollout length; full 140-game
eval ~15-20 min with `--parallel 32`.
