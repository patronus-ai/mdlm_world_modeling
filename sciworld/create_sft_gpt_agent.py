#!/usr/bin/env python3
"""
Generate SFT trajectories using GPT-5.5 as the agent against real ScienceWorld env.
Parallel execution, max 50 turns, filter to score=100 only.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / "trl_training" / ".env")
load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.5"
ENV_URL = "http://localhost:30003"
MAX_TURNS = 50
MAX_WORKERS = 8

GPT_SYSTEM_PROMPT = """You are an expert agent in ScienceWorld, a text-based science experiment environment with 10 rooms.

OUTPUT FORMAT: Output EXACTLY ONE action per turn as plain text. No explanations, no markdown, no numbering — just the action command on a single line.

VALID ACTIONS:
- look around
- teleport to {room}  (always works, use this to move)
- pick up {object}
- put {object} in {container}
- open {object} / close {object}
- activate {object} / deactivate {object}
- focus on {object}  (SUBMITS YOUR ANSWER — wrong focus = -100!)
- pour {object} in {object}
- use {object} on {object}
- mix {object}
- connect {object} to {object}
- wait / wait1
- inventory

ROOMS: hallway, kitchen, workshop, greenhouse, bedroom, living room, art studio, foundry, bathroom, outside

CRITICAL RULES:
- "focus on X" is how you SUBMIT YOUR ANSWER. Only use it when CERTAIN of the correct target.
- For substances in containers: "focus on tin" (the substance), NOT "focus on tin cup" (the container).
- Heat: stove (kitchen), blast furnace (foundry), fire pit (outside). Cold: freezer/fridge (kitchen).
- Open containers (cupboard, freezer, fridge, closet, drawer, oven) to find hidden objects.
- Animal lifespans longest→shortest: giant tortoise > elephant > parrot > dog > wolf > cat > beaver > chipmunk > mouse > dragonfly > bee > ant

STRATEGY: 1) Read task. 2) Teleport to rooms + look around to find objects. 3) Gather needed items. 4) Perform experiment. 5) Submit answer with focus."""

LFM_SYSTEM_PROMPT = """You are a ScienceWorld agent. Output EXACTLY ONE action per turn as plain text. No explanations, no sentences, no markdown — just the action command.

Valid actions: look around, teleport to {room}, pick up {object}, put {object} in {container}, open {object}, close {object}, activate {object}, deactivate {object}, focus on {object}, pour {object} in {object}, use {object} on {object}, mix {object}, connect {object} to {object}, wait, inventory

Rooms: hallway, kitchen, workshop, greenhouse, bedroom, living room, art studio, foundry, bathroom, outside

RULES:
- Always use "teleport to {room}" to move (always works). "go to" often fails.
- "focus on {object}" SUBMITS YOUR ANSWER. Wrong focus = -100 penalty. Only focus when CERTAIN.
- For substances in containers: focus on the SUBSTANCE ("focus on tin"), NOT the container ("focus on tin cup").
- Explore rooms with "teleport to" + "look around". Open containers (cupboard, freezer, fridge, closet) to find hidden objects.
- Heat: stove (kitchen), blast furnace (foundry), fire pit (outside). Cold: freezer, fridge (kitchen).
- Animal lifespans longest→shortest: giant tortoise > elephant > parrot > dog > wolf > cat > beaver > chipmunk > mouse > dragonfly > bee > ant

STRATEGY: 1) Read task. 2) Explore rooms. 3) Gather objects. 4) Do experiment. 5) Submit answer with focus."""


def call_env(task_id, action):
    for attempt in range(2):
        try:
            r = requests.post(ENV_URL, json={"task_id": task_id, "action": action}, timeout=30)
            return r.json()
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                time.sleep(2)
            else:
                return {"observation": "env error", "score": 0.0, "done": True}
    return {"observation": "env error", "score": 0.0, "done": True}


def close_env(task_id):
    try:
        requests.delete(ENV_URL, json={"task_id": task_id}, timeout=5)
    except:
        pass


def parse_gpt_action(text):
    text = text.strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r"^[\d]+[\.\)]\s*", "", text)
    text = re.sub(r"^[-*]\s*", "", text)
    text = text.strip('`"\'')
    text = re.sub(r"^(?:Action|Command|>)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def generate_trajectory(task_row):
    """Use GPT-5.5 as agent against real ScienceWorld env."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    task_id = task_row["task_id"]

    original_msgs = json.loads(task_row["messages"]) if isinstance(task_row["messages"], str) else task_row["messages"]
    user_msg = ""
    for m in original_msgs:
        if m["role"] == "user":
            user_msg = m["content"]
            break

    gpt_messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    sft_trajectory = []

    # Initialize env
    call_env(task_id, "look around")

    score = 0.0
    done = False

    for turn in range(MAX_TURNS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=gpt_messages,
                max_completion_tokens=128,
                extra_body={"reasoning_effort": "none"},
            )
            gpt_output = resp.choices[0].message.content or ""
            gpt_output = gpt_output.strip()
        except Exception as e:
            break

        action = parse_gpt_action(gpt_output)
        if not action:
            break

        sft_trajectory.append({"role": "assistant", "content": action})
        gpt_messages.append({"role": "assistant", "content": action})

        env_resp = call_env(task_id, action)
        obs = env_resp.get("observation", "")
        score = env_resp.get("score", 0.0)
        done = env_resp.get("done", False)

        sft_trajectory.append({"role": "user", "content": obs})
        gpt_messages.append({"role": "user", "content": obs})

        if done:
            break

    close_env(task_id)

    n_actions = len([m for m in sft_trajectory if m["role"] == "assistant"])
    return task_row, sft_trajectory, score, n_actions


def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found")
        sys.exit(1)

    with open("data/sciworld_rl_split.jsonl") as f:
        tasks = [json.loads(l) for l in f]

    print(f"Generating SFT trajectories: {MODEL} (reasoning=none) + real env")
    print(f"Tasks: {len(tasks)}, Workers: {MAX_WORKERS}, Max turns: {MAX_TURNS}")
    print()

    sft_rows = []
    scores = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_trajectory, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            try:
                task_row, traj, score, n_actions = future.result()

                sft_messages = [
                    {"role": "system", "content": LFM_SYSTEM_PROMPT},
                ]
                original_msgs = json.loads(task_row["messages"]) if isinstance(task_row["messages"], str) else task_row["messages"]
                for m in original_msgs:
                    if m["role"] == "user":
                        sft_messages.append({"role": "user", "content": m["content"]})
                        break
                sft_messages.extend(traj)

                icon = "+" if score >= 100 else ("~" if score >= 50 else "-")
                print(f"  [{i+1}/{len(tasks)}] {icon} score={score:.0f} actions={n_actions} | {task_row['instruction'][:60]}")

                scores.append(score)
                sft_rows.append({
                    "messages": sft_messages,
                    "task_id": task_row["task_id"],
                    "instruction": task_row["instruction"],
                    "score": score,
                    "wm_system_prompt": task_row.get("wm_system_prompt", ""),
                })
            except Exception as e:
                print(f"  [{i+1}/{len(tasks)}] ERROR: {e}")

    # Save all
    out_all = "data/sciworld_sft_gpt_all.jsonl"
    with open(out_all, "w") as f:
        for row in sft_rows:
            f.write(json.dumps(row, default=str) + "\n")

    # Filter to score=100 only
    perfect_rows = [r for r in sft_rows if r["score"] >= 100]
    out_perfect = "data/sciworld_sft_gpt_perfect.jsonl"
    with open(out_perfect, "w") as f:
        for row in perfect_rows:
            f.write(json.dumps(row, default=str) + "\n")

    avg_score = sum(scores) / len(scores) if scores else 0
    full = sum(1 for s in scores if s >= 100)
    partial = sum(1 for s in scores if s >= 50)
    print(f"\n{'='*60}")
    print(f"  {MODEL} agent results:")
    print(f"  Total: {len(sft_rows)} trajectories")
    print(f"  Avg score: {avg_score:.1f}")
    print(f"  Full (100): {full}/{len(scores)} ({100*full/len(scores):.0f}%)")
    print(f"  Partial (>=50): {partial}/{len(scores)} ({100*partial/len(scores):.0f}%)")
    print(f"\n  All trajectories: {out_all}")
    print(f"  Perfect (100) only: {out_perfect} ({len(perfect_rows)} rows)")


if __name__ == "__main__":
    main()
