#!/usr/bin/env python3
"""
Generate SFT trajectories using GPT-5.4 as the agent and AppWorld as the ground truth env.
GPT decides what tools to call; AppWorld provides real responses.
"""
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from appworld_prompt import build_agent_system_prompt

load_dotenv(Path.home() / "Downloads" / ".env")
load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5.4"

# Whether to use env server (port 30002) or direct AppWorld import
USE_ENV_SERVER = True
ENV_URL = "http://localhost:30002"

AGENT_SYSTEM_PROMPT = build_agent_system_prompt()


def call_env_server(task_id, tool_name, params):
    """Execute via env server on port 30002."""
    import requests
    r = requests.post(ENV_URL, json={"task_id": task_id, "tool_name": tool_name, "tool_args": params or {}}, timeout=30)
    return r.text


def close_env_server(task_id):
    import requests
    requests.delete(ENV_URL, json={"task_id": task_id}, timeout=5)


def parse_tool_call(text):
    """Parse a tool call from GPT output."""
    try:
        import json_repair
        data = json_repair.loads(text.strip())
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict) and 'name' in data:
            args = data.get('parameters', data.get('arguments', {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}
            return data.get('name'), args if isinstance(args, dict) else {}
    except:
        pass

    # Regex fallback
    m = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if m:
        name = m.group(1)
        args = {}
        pm = re.search(r'"(?:parameters|arguments)"\s*:\s*(\{[^}]*\})', text, re.DOTALL)
        if pm:
            try:
                args = json.loads(pm.group(1))
            except:
                pass
        return name, args
    return None, {}


def generate_trajectory(task_row, client, max_turns=20):
    """Use GPT-5.4 as agent against real AppWorld env."""
    task_id = task_row['task_id']
    instruction = task_row['instruction']

    # Build initial messages
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    trajectory = []  # Only the assistant + tool_response turns

    for turn in range(max_turns):
        # Ask GPT for next action
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=512,
        )
        gpt_output = resp.choices[0].message.content.strip()

        # Parse tool call
        tool_name, tool_args = parse_tool_call(gpt_output)

        if not tool_name:
            break

        # Record assistant turn
        assistant_msg = {"role": "assistant", "content": json.dumps([{"name": tool_name, "parameters": tool_args}])}
        messages.append(assistant_msg)
        trajectory.append(assistant_msg)

        # Check if complete_task
        if 'complete_task' in tool_name:
            break

        # Execute against real env
        env_response = call_env_server(task_id, tool_name, tool_args)

        # Record tool response
        tool_msg = {"role": "user", "content": f"<tool_response>\n{env_response}\n</tool_response>"}
        messages.append(tool_msg)
        trajectory.append(tool_msg)

    close_env_server(task_id)
    return trajectory


def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open('data/appworld_sft_split.jsonl') as f:
        tasks = [json.loads(l) for l in f]

    print(f"Generating SFT trajectories: GPT-5.4 agent + AppWorld env (via env server)")
    print(f"Tasks: {len(tasks)}")

    sft_rows = []
    for i, task in enumerate(tasks):
        try:
            traj = generate_trajectory(task, client)

            # Build SFT sample: system + user + trajectory
            base_messages = json.loads(task['messages']) if isinstance(task['messages'], str) else task['messages']
            sft_messages = base_messages + traj

            # Validate: has show_passwords and complete_task
            full_text = json.dumps(sft_messages)
            has_pwd = 'show_account_passwords' in full_text
            has_complete = 'complete_task' in full_text
            n_turns = len([m for m in traj if m['role'] == 'assistant'])

            # Check for pagination usage
            has_pagination = 'page_limit' in full_text or 'page_index' in full_text

            icon = "✓" if has_pwd and has_complete else "⚠"
            pag = "P" if has_pagination else " "
            print(f"  [{i}] {icon}{pag} {n_turns} turns | pwd={has_pwd} complete={has_complete} | {task['instruction'][:50]}")

            sft_rows.append({
                "messages": sft_messages,
                "task_id": task['task_id'],
                "instruction": task['instruction'],
            })
        except Exception as e:
            print(f"  [{i}] ✗ ERROR: {e} | {task['instruction'][:50]}")

    out_path = 'data/appworld_sft_gpt_agent.jsonl'
    with open(out_path, 'w') as f:
        for row in sft_rows:
            f.write(json.dumps(row, default=str) + '\n')

    n_complete = sum(1 for r in sft_rows if 'complete_task' in json.dumps(r['messages']))
    n_paginated = sum(1 for r in sft_rows if 'page_limit' in json.dumps(r['messages']))
    print(f"\nWrote {out_path}: {len(sft_rows)} demos")
    print(f"  Complete task: {n_complete}/{len(sft_rows)}")
    print(f"  Uses pagination: {n_paginated}/{len(sft_rows)}")


if __name__ == "__main__":
    main()
