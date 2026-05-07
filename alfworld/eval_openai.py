"""Evaluate an OpenAI chat model on the real ALFWorld env.

Usage:
  python eval_openai.py <model> <dataset_jsonl> <out_jsonl> [--max-turns 30] [--limit N]
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

from alfworld_prompt import format_user_turn

ENV_ENDPOINT = os.environ.get("ALFWORLD_ENDPOINT", "http://localhost:30003")
ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

LOOP_REPEAT_LIMIT    = int(os.environ.get("ALFWORLD_LOOP_LIMIT", "3"))
NOTHING_STREAK_LIMIT = int(os.environ.get("ALFWORLD_NOTHING_LIMIT", "4"))


def parse_action(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = ACTION_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or None


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


def run_episode(client, model, row, max_turns):
    task_id = f"oai-{uuid.uuid4().hex[:8]}"
    reset = env_reset(task_id, row["game_file"], row.get("split", "eval_in_distribution"))
    messages = [
        {"role": "system", "content": row["messages"][0]["content"]},
        {"role": "user", "content": format_user_turn(reset["obs"], reset["admissible"])},
    ]
    raw_completions = []
    parsed_actions = []
    won = False
    turns = 0
    kill = None
    recent_actions = []
    nothing_streak = 0

    try:
        for turn in range(max_turns):
            turns = turn + 1
            try:
                resp = client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=256,
                    reasoning={"effort": "none"},
                )
                completion = resp.output_text or ""
            except Exception as e:
                kill = f"api_error:{e}"; break

            messages.append({"role": "assistant", "content": completion})
            raw_completions.append(completion)
            action = parse_action(completion)
            parsed_actions.append(action)
            if not action:
                kill = "unparseable"; break
            recent_actions.append(action.lower())
            if (len(recent_actions) >= LOOP_REPEAT_LIMIT
                    and len(set(recent_actions[-LOOP_REPEAT_LIMIT:])) == 1):
                kill = "loop"; break

            step = env_step(task_id, action)
            if "error" in step:
                kill = f"env_error:{step['error']}"; break
            won = bool(step.get("won")); done = bool(step.get("done"))
            obs = step["obs"]
            if obs.strip() == "Nothing happens.":
                nothing_streak += 1
            else:
                nothing_streak = 0
            messages.append({"role": "user",
                             "content": format_user_turn(obs, step["admissible"][:30])})
            if won or done:
                break
            if nothing_streak >= NOTHING_STREAK_LIMIT:
                kill = "stuck"; break
    finally:
        env_close(task_id)

    return {
        "won": won, "turns": turns, "kill": kill,
        "messages": messages, "raw_completions": raw_completions,
        "parsed_actions": parsed_actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    client = OpenAI()
    rows = [json.loads(l) for l in open(args.dataset)]
    if args.limit > 0:
        rows = rows[:args.limit]

    n_won = 0
    t0 = time.time()
    results: dict[int, dict] = {}

    with open(args.out, "w") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {ex.submit(run_episode, client, args.model, row, args.max_turns): (i, row)
                       for i, row in enumerate(rows)}
            done = 0
            for fut in as_completed(futures):
                i, row = futures[fut]
                try:
                    ep = fut.result()
                except Exception as e:
                    ep = {"won": False, "turns": 0, "kill": f"exception:{e}",
                          "messages": [], "raw_completions": [], "parsed_actions": []}
                ep["i"] = i
                ep["task_type"] = row.get("task_type")
                ep["goal"] = row.get("messages", [{}])[1].get("content", "").split("Your task is to:")[-1].split("\n")[0].strip()[:120]
                ep["game_file"] = row["game_file"]
                results[i] = ep
                done += 1
                if ep["won"]:
                    n_won += 1
                print(f"[{done}/{len(rows)}] i={i} won={ep['won']} turns={ep['turns']} "
                      f"kill={ep.get('kill')} task={row.get('task_type')} acc={n_won/done:.2f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        # write in deterministic order
        for i in sorted(results):
            f.write(json.dumps(results[i]) + "\n")

    print(f"\nFinal: {n_won}/{len(rows)} = {n_won/max(len(rows),1):.3f}")
    print(f"trajectories -> {args.out}")


if __name__ == "__main__":
    main()
