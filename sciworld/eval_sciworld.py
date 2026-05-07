#!/usr/bin/env python3
"""Evaluate model on ScienceWorld tasks against real environment."""
import json, os, re, sys, requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sciworld_wm_prompt import parse_action

ENV_URL = "http://localhost:30004"

def call_env(task_id, action):
    for attempt in range(2):
        try:
            r = requests.post(ENV_URL, json={"task_id": task_id, "action": action}, timeout=30)
            return r.json()
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                import time; time.sleep(2)
            else:
                return {"observation": "env connection failed", "score": 0.0, "done": True}
    return {"observation": "env connection failed", "score": 0.0, "done": True}

def close_env(task_id):
    try:
        requests.delete(ENV_URL, json={"task_id": task_id}, timeout=5)
    except:
        pass

def run_episode(llm, tokenizer, task, max_turns=50):
    task_id = task["task_id"]
    close_env(task_id)
    messages = json.loads(task["messages"]) if isinstance(task["messages"], str) else list(task["messages"])
    params = SamplingParams(temperature=0.0, max_tokens=256, stop=["<|im_end|>", "<|endoftext|>", "</s>"])

    # Initialize env — first POST creates the env AND executes "look around"
    init_resp = call_env(task_id, "look around")
    if "error" in init_resp:
        return {"score": 0.0, "n_steps": 0, "done": False, "error": init_resp.get("error", "")}

    all_actions = []
    final_score = 0.0
    done = False

    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tok_len = len(tokenizer.encode(prompt))
        if tok_len > 15000:
            break

        output = llm.generate([prompt], params)
        completion = output[0].outputs[0].text.strip()
        action = parse_action(completion)

        if not action:
            break

        all_actions.append(action)
        resp = call_env(task_id, action)
        obs = resp.get("observation", "")
        final_score = resp.get("score", 0.0)
        done = resp.get("done", False)

        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": obs})

        if done:
            break

    close_env(task_id)

    n_repeated = 0
    for i in range(2, len(all_actions)):
        if all_actions[i] == all_actions[i-1] == all_actions[i-2]:
            n_repeated += 1

    return {
        "score": final_score,
        "n_steps": len(all_actions),
        "done": done,
        "n_repeated": n_repeated,
        "last_action": all_actions[-1] if all_actions else "",
    }

def main():
    model_path = sys.argv[1]
    test_file = sys.argv[2] if len(sys.argv) > 2 else "data/sciworld_test_split.jsonl"
    label = sys.argv[3] if len(sys.argv) > 3 else "model"
    gpu = sys.argv[4] if len(sys.argv) > 4 else "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print(f"Model: {model_path}\nTest: {test_file}\nGPU: {gpu}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, gpu_memory_utilization=0.85, max_model_len=16384, trust_remote_code=True)

    with open(test_file) as f:
        tasks = [json.loads(l) for l in f]
    print(f"\nRunning {len(tasks)} ScienceWorld episodes...\n")

    results = []
    for i, task in enumerate(tasks):
        try:
            r = run_episode(llm, tokenizer, task)
            r["task_id"] = task["task_id"]
            r["instruction"] = task["instruction"][:60]
            results.append(r)
            s = "+" if r["score"] >= 50 else "-"
            print(f"  [{i+1}/{len(tasks)}] {s} score={r['score']:.1f} steps={r['n_steps']} done={r['done']} | {r['instruction']}")
        except Exception as e:
            print(f"  [{i+1}/{len(tasks)}] ERROR: {e}")
            results.append({"score": 0.0, "n_steps": 0, "done": False, "task_id": task["task_id"], "instruction": task["instruction"][:60]})

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)
    completed = sum(1 for r in results if r["score"] >= 100) / len(results)
    partial = sum(1 for r in results if r["score"] >= 50) / len(results)
    avg_steps = sum(r["n_steps"] for r in results) / len(results)

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  Average score:     {avg_score:.1f}/100")
    print(f"  Full completion:   {completed:.1%}")
    print(f"  Partial (>=50):    {partial:.1%}")
    print(f"  Avg steps:         {avg_steps:.1f}")

    by_type = {}
    for r in results:
        tt = r["task_id"].rsplit("_v", 1)[0]
        by_type.setdefault(tt, []).append(r["score"])
    print(f"\n  Per-task-type scores:")
    for tt in sorted(by_type.keys()):
        s = by_type[tt]
        print(f"    {tt}: {sum(s)/len(s):.1f} ({len(s)} tasks)")

    with open(f"data/eval_{label}.json", "w") as f:
        json.dump({"label": label, "avg_score": avg_score, "results": results}, f, indent=2)
    print(f"\n  Saved: data/eval_{label}.json")

if __name__ == "__main__":
    main()
