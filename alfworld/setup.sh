#!/bin/bash
# ALFWorld Training Pipeline — Full Setup Script
# Run this on a fresh cluster node with 8x80GB GPUs, CUDA 12.x, Python 3.11
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== ALFWorld Training Pipeline Setup ==="

# ---------------------------------------------------------------
# 1. Create venv
# ---------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[1/7] Creating Python 3.11 venv..."
    python3.11 -m venv .venv
else
    echo "[1/7] venv already exists"
fi
source .venv/bin/activate

# ---------------------------------------------------------------
# 2. Install dependencies (order matters for compatibility)
# ---------------------------------------------------------------
echo "[2/7] Installing dependencies..."

# Use uv for fast installs if available, else pip
if command -v uv &>/dev/null; then
    PIP="uv pip install"
else
    PIP="pip install"
fi

# PyTorch (adjust cu version for your CUDA)
$PIP torch --index-url https://download.pytorch.org/whl/cu124

# ALFWorld + tatsu pin (textworld grammar parser breaks on newer tatsu)
$PIP alfworld "tatsu==5.8.3"

# vLLM for inference + GRPO rollouts
$PIP vllm

# ms-swift (training framework) — clone and install editable
if [ ! -d "ms-swift" ]; then
    git clone https://github.com/modelscope/ms-swift.git
    cd ms-swift && git checkout 43b5d8e && cd ..
fi
$PIP -e ms-swift

# Other deps
$PIP transformers requests python-dotenv openai wandb hf_transfer json_repair lmdeploy

echo "  Dependencies installed."

# ---------------------------------------------------------------
# 3. ALFWorld data download (~315MB)
# ---------------------------------------------------------------
echo "[3/7] Downloading ALFWorld data..."
export ALFWORLD_DATA="$SCRIPT_DIR/data"
if [ ! -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    alfworld-download
else
    echo "  ALFWorld data already present."
fi

# ---------------------------------------------------------------
# 4. Download model weights
# ---------------------------------------------------------------
echo "[4/7] Downloading model weights..."
export HF_HUB_ENABLE_HF_TRANSFER=1
export USE_HF=1

# Read HF token from .env if present
if [ -f .env ]; then
    HF_TOKEN=$(grep HF_TOKEN .env 2>/dev/null | cut -d= -f2 | tr -d '"')
fi
TOKEN_FLAG=""
if [ -n "$HF_TOKEN" ]; then
    TOKEN_FLAG="--token $HF_TOKEN"
fi

# Mistral-7B (agent model)
echo "  Downloading Mistral-7B-Instruct-v0.3..."
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 $TOKEN_FLAG || echo "  (may need HF_TOKEN for gated model)"

# World Model — Qwen3.5-35B-A3B based (MoE, 35B total / 3B active)
echo "  Downloading Qwen3.5 World Model..."
WM_DIR="${HF_HOME:-$HOME/.cache/huggingface}/qwen35_wm_v2"
if [ ! -d "$WM_DIR" ] || [ ! -f "$WM_DIR/config.json" ]; then
    mkdir -p "$WM_DIR"
    huggingface-cli download ANONYMOUS/Qwen3.5-35BA3B_world_model_v2 --local-dir "$WM_DIR" $TOKEN_FLAG
fi

# Also download SDAR WM v2 (diffusion LM, alternative WM)
echo "  Downloading SDAR World Model v2..."
huggingface-cli download ANONYMOUS/SDAR_world_model_v2 $TOKEN_FLAG || true

# ---------------------------------------------------------------
# 5. Patch model configs
# ---------------------------------------------------------------
echo "[5/7] Patching model configs..."

# SDAR: add missing pad_token_id
SDAR_DIR=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ANONYMOUS--SDAR_world_model_v2/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$SDAR_DIR" ] && [ -f "${SDAR_DIR}config.json" ]; then
    python3 -c "
import json
p = '${SDAR_DIR}config.json'
d = json.load(open(p))
if d.get('pad_token_id') is None:
    d['pad_token_id'] = 151643
    json.dump(d, open(p, 'w'), indent=2)
    print('  Patched SDAR pad_token_id')
else:
    print('  SDAR config already patched')
"
fi

# Qwen3.5 WM: fix tokenizer_class if needed
if [ -f "$WM_DIR/tokenizer_config.json" ]; then
    python3 -c "
import json
p = '$WM_DIR/tokenizer_config.json'
d = json.load(open(p))
if d.get('tokenizer_class') == 'TokenizersBackend':
    # Replace with base model tokenizer config
    import subprocess
    subprocess.run(['curl', '-sL', '-H', 'Authorization: Bearer ${HF_TOKEN}',
        'https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main/tokenizer_config.json',
        '-o', p], check=True)
    print('  Fixed Qwen3.5 WM tokenizer_config')
else:
    print('  Qwen3.5 WM tokenizer OK')
"
fi

# Mistral: ensure HF-format weights exist (not just consolidated.safetensors)
MISTRAL_DIR=$(ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/ 2>/dev/null | head -1)
if [ -n "$MISTRAL_DIR" ] && [ ! -f "${MISTRAL_DIR}model-00001-of-00003.safetensors" ]; then
    echo "  WARNING: Mistral weights may be in native format. Run: huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3"
fi

# ---------------------------------------------------------------
# 6. Build datasets
# ---------------------------------------------------------------
echo "[6/7] Building datasets..."
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Eval dataset (140 in-distribution games)
if [ ! -f "data/alfworld_eval_id.jsonl" ]; then
    python3 -c "
import sys; sys.setrecursionlimit(50000)
sys.argv = ['build_dataset.py', '--split', 'eval_in_distribution', '--out', 'data/alfworld_eval_id.jsonl']
from build_dataset import _alfred, _run_loop, format_user_turn, AGENT_SYSTEM_PROMPT
alfred = _alfred('eval_in_distribution')
games = sorted(alfred.game_files)
print(f'Building eval set: {len(games)} games')
_run_loop(games, 'eval_in_distribution', 'data/alfworld_eval_id.jsonl')
"
else
    echo "  Eval dataset already exists"
fi

# RL training dataset (300 shuffled train games)
if [ ! -f "data/alfworld_rl.jsonl" ]; then
    python3 -c "
import sys; sys.setrecursionlimit(50000)
import random
sys.argv = ['build_dataset.py', '--split', 'train', '--out', 'data/alfworld_rl.jsonl', '--limit', '300', '--shuffle']
from build_dataset import _alfred, _run_loop
alfred = _alfred('train')
games = sorted(alfred.game_files)
random.Random(7).shuffle(games)
games = games[:300]
print(f'Building RL set: {len(games)} games')
_run_loop(games, 'train', 'data/alfworld_rl.jsonl')
"
else
    echo "  RL dataset already exists"
fi

# SFT expert trajectories
if [ ! -f "data/alfworld_sft_large.jsonl" ]; then
    echo "  Building SFT expert data (this takes ~10min)..."
    python3 -c "
import sys; sys.setrecursionlimit(50000)
import json, random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
mp.set_start_method('fork', force=True)
from build_sft_expert import _alfred, rollout_expert

alfred = _alfred()
games = sorted(alfred.game_files)
random.Random(42).shuffle(games)
type_counts = Counter()
selected = []
for gf in games:
    for tt in ['look_at_obj_in_light', 'pick_and_place_simple', 'pick_clean_then_place_in_recep',
               'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_two_obj_and_place']:
        if tt in gf:
            if type_counts[tt] < 200:
                selected.append(gf)
                type_counts[tt] += 1
            break
    if sum(type_counts.values()) >= 1200:
        break
print(f'Processing {len(selected)} games with 8 workers')

def process_game(gf):
    try:
        sys.setrecursionlimit(50000)
        messages, won = rollout_expert(gf)
        if won: return json.dumps({'messages': messages, 'game_file': gf})
    except: pass
    return None

results = []
with ProcessPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(process_game, gf): gf for gf in selected}
    for i, f in enumerate(as_completed(futures)):
        r = f.result()
        if r: results.append(r)
        if (i+1) % 100 == 0: print(f'  {i+1}/{len(selected)} kept={len(results)}')
with open('data/alfworld_sft_large.jsonl', 'w') as f:
    for r in results: f.write(r + '\n')
print(f'Wrote {len(results)} expert trajectories')
"
else
    echo "  SFT dataset already exists"
fi

# Create Mistral-formatted versions (system prompt merged into first user message)
python3 -c "
import json, os
for src, dst in [('data/alfworld_sft_large.jsonl', 'data/alfworld_sft_large_mistral.jsonl'),
                 ('data/alfworld_rl.jsonl', 'data/alfworld_rl_mistral.jsonl')]:
    if not os.path.exists(src): continue
    if os.path.exists(dst):
        print(f'  {dst} already exists'); continue
    rows = []
    for line in open(src):
        try: rows.append(json.loads(line))
        except: pass
    out = []
    for row in rows:
        msgs = row.get('messages', [])
        if not msgs or msgs[0].get('role') != 'system':
            out.append(row); continue
        sys_c = msgs[0]['content']
        new_msgs = []
        found = False
        for m in msgs[1:]:
            if m['role'] == 'user' and not found:
                new_msgs.append({'role': 'user', 'content': sys_c + '\n\n' + m['content']})
                found = True
            else:
                new_msgs.append(m)
        new_row = dict(row)
        new_row['messages'] = new_msgs
        out.append(new_row)
    with open(dst, 'w') as f:
        for r in out: f.write(json.dumps(r) + '\n')
    print(f'  Created {dst} ({len(out)} rows)')
"

# ---------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "Files created:"
for f in data/alfworld_eval_id.jsonl data/alfworld_rl.jsonl data/alfworld_rl_mistral.jsonl data/alfworld_sft_large.jsonl data/alfworld_sft_large_mistral.jsonl; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(wc -l < $f) rows)"
    else
        echo "  ✗ $f (missing)"
    fi
done
echo ""
echo "Next steps:"
echo "  1. Create .env with WANDB_API_KEY, HF_TOKEN, OPENAI_API_KEY"
echo "  2. Start WM:    See GETTING_STARTED.md §4"
echo "  3. Start env:   python alfworld_env_server.py &"
echo "  4. SFT:         bash run_mistral_sft.sh"
echo "  5. GRPO:        bash run_mistral_grpo_2gpus.sh"
echo "  6. Eval:        python eval_fast.py <checkpoint> --dataset data/alfworld_eval_id.jsonl --gpus 0"
