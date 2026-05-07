import sys; sys.setrecursionlimit(50000)  # spacy/textworld deep import chain
"""Parallel ALFWorld eval: advance N episodes in lockstep, batching generate calls
through vLLM. Env server serialises step calls via its lock, but a single batched
llm.generate(prompts) call amortises GPU time across all active episodes.

Usage:
  python eval_parallel.py <model> <dataset_jsonl> <out_jsonl> <gpu_id>
                          [--max-turns 50] [--limit N] [--parallel 32]
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

ENV_ENDPOINT = os.environ.get("ALFWORLD_ENDPOINT", "http://localhost:30003")
ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
LOOP_REPEAT_LIMIT    = int(os.environ.get("ALFWORLD_LOOP_LIMIT", "3"))
NOTHING_STREAK_LIMIT = int(os.environ.get("ALFWORLD_NOTHING_LIMIT", "4"))


def parse_action(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = ACTION_RE.search(text)
    action = m.group(1).strip().rstrip(".").strip() if m else (text.strip().splitlines()[0].strip() if text.strip() else "")
    if not action:
        return None
    action = re.sub(r"\b(?:the|a|an) ([\w]+ \d+)", r"\1", action, flags=re.I)
    return action


def _post_with_retry(url, payload, timeout, max_retries=5, method="POST"):
    """Retry on transient connection errors (env server drops conns under load)."""
    last = None
    for i in range(max_retries):
        try:
            if method == "POST":
                r = requests.post(url, json=payload, timeout=timeout)
            else:
                r = requests.delete(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            last = e
            time.sleep(0.2 * (i + 1))
    raise last


def env_reset(task_id, game_file, split):
    return _post_with_retry(f"{ENV_ENDPOINT}/reset",
                            {"task_id": task_id, "game_file": game_file, "split": split},
                            timeout=120, max_retries=5)


def env_step(task_id, action):
    return _post_with_retry(f"{ENV_ENDPOINT}/step",
                            {"task_id": task_id, "action": action},
                            timeout=60, max_retries=5)


def env_close(task_id):
    try:
        _post_with_retry(f"{ENV_ENDPOINT}/close", {"task_id": task_id},
                         timeout=10, max_retries=3, method="DELETE")
    except Exception:
        pass


class Episode:
    """One in-flight rollout: messages, env task_id, termination state."""

    def __init__(self, idx: int, row: dict, system_prompt: str, format_user_turn):
        self.idx = idx
        self.row = row
        self.task_id = f"par-{uuid.uuid4().hex[:10]}"
        self.format_user_turn = format_user_turn
        self.messages: list[dict] = []
        self.parsed_actions: list[str] = []
        self.raw_completions: list[str] = []
        self.recent_actions: list[str] = []
        self.nothing_streak = 0
        self.won = False
        self.done = False
        self.kill: str | None = None
        self.turns = 0
        self.system_prompt = system_prompt

    def reset(self):
        try:
            r = env_reset(self.task_id, self.row["game_file"], self.row.get("split", "eval_in_distribution"))
        except Exception as e:
            self.kill = f"reset_error:{e}"; self.done = True; return
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.format_user_turn(r["obs"], r["admissible"])},
        ]

    def post_step(self, completion: str):
        self.turns += 1
        self.messages.append({"role": "assistant", "content": completion})
        self.raw_completions.append(completion)
        a = parse_action(completion)
        self.parsed_actions.append(a)
        if not a:
            self.kill = "unparseable"; self.done = True; return
        self.recent_actions.append(a.lower())
        if (len(self.recent_actions) >= LOOP_REPEAT_LIMIT
                and len(set(self.recent_actions[-LOOP_REPEAT_LIMIT:])) == 1):
            self.kill = "loop"; self.done = True; return
        try:
            step = env_step(self.task_id, a)
        except Exception as e:
            self.kill = f"env_error:{e}"; self.done = True; return
        if "error" in step:
            self.kill = f"env_error:{step['error']}"; self.done = True; return
        self.won = bool(step.get("won")); env_done = bool(step.get("done"))
        obs = step["obs"]
        if obs.strip() == "Nothing happens.":
            self.nothing_streak += 1
        else:
            self.nothing_streak = 0
        self.messages.append({
            "role": "user",
            "content": self.format_user_turn(obs, step["admissible"][:30]),
        })
        if self.won or env_done:
            self.done = True; return
        if self.nothing_streak >= NOTHING_STREAK_LIMIT:
            self.kill = "stuck"; self.done = True; return

    def result(self) -> dict:
        return {
            "i": self.idx, "won": self.won, "turns": self.turns, "kill": self.kill,
            "task_type": self.row.get("task_type"),
            "goal": self.row.get("messages", [{}])[1].get("content", "").split("Your task is to:")[-1].split("\n")[0].strip()[:120],
            "game_file": self.row["game_file"],
            "messages": self.messages,
            "raw_completions": self.raw_completions,
            "parsed_actions": self.parsed_actions,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("gpu_id")
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from alfworld_prompt import format_user_turn

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, gpu_memory_utilization=0.85,
              max_model_len=32768, dtype="bfloat16", trust_remote_code=True)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=128,
                              stop=["<|im_end|>", "<|endoftext|>", "</s>"])

    rows = [json.loads(l) for l in open(args.dataset)]
    if args.limit > 0:
        rows = rows[:args.limit]
    system_prompt = rows[0]["messages"][0]["content"]

    # Reset all episodes in parallel (env server serialises but parallel HTTP saves wall time)
    episodes = [Episode(i, r, system_prompt, format_user_turn) for i, r in enumerate(rows)]
    t0 = time.time()
    print(f"resetting {len(episodes)} episodes...", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(lambda e: e.reset(), episodes))
    print(f"reset done in {time.time()-t0:.0f}s", flush=True)

    n_done = 0
    n_won = 0
    turn_idx = 0
    while True:
        active = [e for e in episodes if not e.done and e.turns < args.max_turns]
        if not active:
            break
        # batch of prompts (one per active episode)
        prompts = [tokenizer.apply_chat_template(e.messages, tokenize=False, add_generation_prompt=True)
                   for e in active]
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        # post_step in parallel — env server serialises /step
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            list(ex.map(lambda pair: pair[0].post_step(pair[1].outputs[0].text),
                        zip(active, outs)))
        # bookkeeping
        new_done = [e for e in active if e.done]
        for e in new_done:
            n_won += int(e.won)
        n_done += len(new_done)
        turn_idx += 1
        elapsed = time.time() - t0
        live = sum(1 for e in episodes if not e.done and e.turns < args.max_turns)
        print(f"turn {turn_idx:2}: active={len(active)} live={live} done={n_done} won={n_won} acc={n_won/max(n_done,1):.3f} elapsed={elapsed:.0f}s",
              flush=True)
        if turn_idx >= args.max_turns:
            for e in episodes:
                if not e.done:
                    e.kill = e.kill or "turn_cap"
                    e.done = True
            break

    # close all envs
    print("closing envs...", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(lambda e: env_close(e.task_id), episodes))

    # dump results in input order
    with open(args.out, "w") as f:
        for e in episodes:
            f.write(json.dumps(e.result()) + "\n")
    n_won_final = sum(1 for e in episodes if e.won)
    print(f"\nFinal: {n_won_final}/{len(episodes)} = {n_won_final/max(len(episodes),1):.3f}")
    print(f"results -> {args.out}")
    print(f"total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
