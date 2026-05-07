"""Evaluate a model on ALFWorld via vLLM + the local env server.

Usage:
  python eval.py <model_path> <dataset_jsonl> <label> <gpu_id> [--max-turns 30]
"""
import argparse
import json
import os
import re
import sys
import time
import uuid

import requests

ENV_ENDPOINT = os.environ.get("ALFWORLD_ENDPOINT", "http://localhost:30003")
ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

LOOP_REPEAT_LIMIT    = int(os.environ.get("ALFWORLD_LOOP_LIMIT", "3"))
NOTHING_STREAK_LIMIT = int(os.environ.get("ALFWORLD_NOTHING_LIMIT", "4"))


def parse_action(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = ACTION_RE.search(text)
    if m:
        action = m.group(1).strip().rstrip(".").strip()
    else:
        action = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not action:
        return None
    # textworld rejects "take the bowl 1"; normalize to "take bowl 1".
    action = re.sub(r"\b(?:the|a|an) ([\w]+ \d+)", r"\1", action, flags=re.I)
    return action


def env_reset(task_id, game_file, split):
    return requests.post(f"{ENV_ENDPOINT}/reset",
                         json={"task_id": task_id, "game_file": game_file, "split": split},
                         timeout=120).json()


def env_step(task_id, action):
    return requests.post(f"{ENV_ENDPOINT}/step",
                         json={"task_id": task_id, "action": action}, timeout=60).json()


def env_close(task_id):
    try:
        requests.delete(f"{ENV_ENDPOINT}/close", json={"task_id": task_id}, timeout=10)
    except Exception:
        pass


def run_episode(llm, tokenizer, sampling_params, row, max_turns: int):
    from alfworld_prompt import format_user_turn

    task_id = f"eval-{uuid.uuid4().hex[:8]}"
    reset = env_reset(task_id, row["game_file"], row.get("split", "eval_in_distribution"))
    messages = [
        {"role": "system", "content": row["messages"][0]["content"]},
        {"role": "user", "content": format_user_turn(reset["obs"], reset["admissible"])},
    ]
    won = False
    turns = 0
    kill = None
    recent_actions: list[str] = []
    nothing_streak = 0
    try:
        for turn in range(max_turns):
            turns = turn + 1
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
            completion = out.outputs[0].text
            messages.append({"role": "assistant", "content": completion})
            action = parse_action(completion)
            if not action:
                kill = "unparseable"; break
            recent_actions.append(action.lower())
            if (len(recent_actions) >= LOOP_REPEAT_LIMIT
                    and len(set(recent_actions[-LOOP_REPEAT_LIMIT:])) == 1):
                kill = "loop"; break
            step = env_step(task_id, action)
            if "error" in step:
                kill = "env_error"; break
            won = bool(step.get("won"))
            done = bool(step.get("done"))
            obs = step["obs"]
            if obs.strip() == "Nothing happens.":
                nothing_streak += 1
            else:
                nothing_streak = 0
            messages.append({
                "role": "user",
                "content": format_user_turn(obs, step["admissible"][:30]),
            })
            if won or done:
                break
            if nothing_streak >= NOTHING_STREAK_LIMIT:
                kill = "stuck"; break
    finally:
        env_close(task_id)
    return won, turns, kill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("dataset")
    ap.add_argument("label")
    ap.add_argument("gpu_id")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.85,
              max_model_len=16384, dtype="bfloat16")
    sampling = SamplingParams(temperature=0.0, max_tokens=128, stop=["<|im_end|>"])

    rows = [json.loads(l) for l in open(args.dataset)]
    if args.limit > 0:
        rows = rows[:args.limit]

    out_path = f"/tmp/alfworld_eval_{args.label}.jsonl"
    n_won = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, row in enumerate(rows):
            try:
                won, turns, kill = run_episode(llm, tokenizer, sampling, row, args.max_turns)
            except Exception as e:
                won, turns, kill = False, 0, f"exception:{e}"
                print(f"[{i}] error: {e}", file=sys.stderr)
            n_won += int(won)
            f.write(json.dumps({"i": i, "game_file": row["game_file"],
                                "won": won, "turns": turns, "kill": kill}) + "\n")
            f.flush()
            print(f"[{i+1}/{len(rows)}] won={won} turns={turns} kill={kill} "
                  f"acc={n_won/(i+1):.3f} elapsed={(time.time()-t0):.0f}s")

    print(f"\n{args.label}: {n_won}/{len(rows)} = {n_won/max(len(rows),1):.3f}")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
