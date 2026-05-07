#!/usr/bin/env python3
"""Fast ALFWorld eval: load model ONCE, eval multiple checkpoints.
Uses tp=2 for faster generation, lower max_tokens, TMPDIR on /workspace.

Usage:
  python eval_fast.py <ckpt1> [ckpt2] ... --dataset data/alfworld_eval_id.jsonl --gpus 0,3
"""
import sys; sys.setrecursionlimit(50000)
import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

os.environ.setdefault("TMPDIR", "/workspace/tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

ENV_ENDPOINT = os.environ.get("ALFWORLD_ENDPOINT", "http://localhost:30003")
ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
LOOP_LIMIT = 3
NOTHING_LIMIT = 4


def parse_action(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = ACTION_RE.search(text)
    action = m.group(1).strip().rstrip(".").strip() if m else (text.strip().splitlines()[0].strip() if text.strip() else "")
    if not action:
        return None
    action = action[0].lower() + action[1:] if action else action
    action = re.sub(r"\b(?:the|a|an) ([\w]+ \d+)", r"\1", action, flags=re.I)
    return action


def _post(url, payload, timeout=60, retries=3):
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(0.5)


class Episode:
    def __init__(self, idx, row, system_prompt, format_user_turn):
        self.idx = idx
        self.row = row
        self.task_id = f"fe-{uuid.uuid4().hex[:8]}"
        self.format_user_turn = format_user_turn
        self.messages = []
        self.recent = []
        self.nothing_streak = 0
        self.won = False
        self.done = False
        self.kill = None
        self.turns = 0
        self.system_prompt = system_prompt
        self.actions = []

    def reset(self):
        try:
            r = _post(f"{ENV_ENDPOINT}/reset",
                       {"task_id": self.task_id, "game_file": self.row["game_file"],
                        "split": self.row.get("split", "eval_in_distribution")}, timeout=120)
        except Exception as e:
            self.kill = f"reset_error:{e}"; self.done = True; return
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.format_user_turn(r["obs"], r["admissible"])},
        ]

    def step(self, completion):
        self.turns += 1
        self.messages.append({"role": "assistant", "content": completion})
        a = parse_action(completion)
        self.actions.append(a)
        if not a:
            self.kill = "unparseable"; self.done = True; return
        self.recent.append(a.lower())
        # Loop detection: AAA
        if len(self.recent) >= LOOP_LIMIT and len(set(self.recent[-LOOP_LIMIT:])) == 1:
            self.kill = "loop"; self.done = True; return
        # ABAB
        if len(self.recent) >= 4 and self.recent[-1] == self.recent[-3] and self.recent[-2] == self.recent[-4] and self.recent[-1] != self.recent[-2]:
            self.kill = "loop_abab"; self.done = True; return
        try:
            s = _post(f"{ENV_ENDPOINT}/step", {"task_id": self.task_id, "action": a})
        except Exception as e:
            self.kill = f"env_error:{e}"; self.done = True; return
        if "error" in s:
            self.kill = f"env_error:{s['error']}"; self.done = True; return
        self.won = bool(s.get("won"))
        obs = s["obs"]
        if obs.strip() == "Nothing happens.":
            self.nothing_streak += 1
        else:
            self.nothing_streak = 0
        self.messages.append({"role": "user", "content": self.format_user_turn(obs, s["admissible"][:30])})
        if self.won or s.get("done"):
            self.done = True; return
        if self.nothing_streak >= NOTHING_LIMIT:
            self.kill = "stuck"; self.done = True

    def close(self):
        try:
            requests.delete(f"{ENV_ENDPOINT}/close", json={"task_id": self.task_id}, timeout=10)
        except:
            pass

    def result(self):
        return {"i": self.idx, "won": self.won, "turns": self.turns, "kill": self.kill,
                "task_type": self.row.get("task_type"), "game_file": self.row["game_file"]}


def eval_checkpoint(llm, tokenizer, sampling, rows, system_prompt, format_user_turn, max_turns, parallel):
    episodes = [Episode(i, r, system_prompt, format_user_turn) for i, r in enumerate(rows)]
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(lambda e: e.reset(), episodes))

    for turn_idx in range(max_turns):
        active = [e for e in episodes if not e.done and e.turns < max_turns]
        if not active:
            break
        prompts = [tokenizer.apply_chat_template(e.messages, tokenize=False, add_generation_prompt=True)
                   for e in active]
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            list(ex.map(lambda pair: pair[0].step(pair[1].outputs[0].text),
                        zip(active, outs)))
        n_done = sum(1 for e in episodes if e.done)
        n_won = sum(1 for e in episodes if e.won)
        live = sum(1 for e in episodes if not e.done)
        if turn_idx % 5 == 0 or live == 0:
            print(f"  turn {turn_idx+1:2}: live={live} done={n_done} won={n_won} [{time.time()-t0:.0f}s]", flush=True)

    for e in episodes:
        if not e.done:
            e.kill = "turn_cap"; e.done = True

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(lambda e: e.close(), episodes))

    won = sum(1 for e in episodes if e.won)
    elapsed = time.time() - t0
    return won, len(episodes), elapsed, [e.result() for e in episodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpus", default="0,3")
    ap.add_argument("--max-turns", type=int, default=35)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from alfworld_prompt import format_user_turn

    rows = [json.loads(l) for l in open(args.dataset)]
    system_prompt = rows[0]["messages"][0]["content"]
    sampling = SamplingParams(temperature=0.0, max_tokens=32,
                              stop=["<|im_end|>", "<|endoftext|>", "</s>", "\n"])

    # Load model ONCE with tp=number of GPUs
    n_gpus = len(args.gpus.split(","))
    print(f"Loading model from {args.checkpoints[0]} on {n_gpus} GPUs...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints[0], trust_remote_code=True)
    llm = LLM(model=args.checkpoints[0], gpu_memory_utilization=0.85,
              max_model_len=16384, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=n_gpus)
    print("Model loaded.", flush=True)

    for ckpt in args.checkpoints:
        if ckpt != args.checkpoints[0]:
            # Swap weights for subsequent checkpoints
            # vLLM doesn't support hot-swap, so we need to reload
            print(f"\nReloading model from {ckpt}...", flush=True)
            del llm
            import gc; gc.collect()
            import torch; torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
            llm = LLM(model=ckpt, gpu_memory_utilization=0.85,
                      max_model_len=16384, dtype="bfloat16", trust_remote_code=True,
                      tensor_parallel_size=n_gpus)

        label = os.path.basename(ckpt)
        print(f"\n{'='*60}\nEval: {ckpt}\n{'='*60}", flush=True)
        won, total, elapsed, results = eval_checkpoint(
            llm, tokenizer, sampling, rows, system_prompt, format_user_turn,
            args.max_turns, args.parallel)

        out_path = os.path.join(args.out_dir, f"eval_{label}.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        from collections import Counter
        by_type = {}
        for r in results:
            tt = r.get("task_type", "?")
            by_type.setdefault(tt, {"w": 0, "t": 0})
            by_type[tt]["t"] += 1
            if r["won"]: by_type[tt]["w"] += 1
        kills = Counter(r.get("kill") or "completed" for r in results if not r["won"])

        print(f"\nResult: {won}/{total} = {won/total:.1%} in {elapsed:.0f}s")
        for tt in sorted(by_type):
            print(f"  {tt:<40} {by_type[tt]['w']}/{by_type[tt]['t']}")
        print(f"  Kills: {dict(kills.most_common(5))}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
