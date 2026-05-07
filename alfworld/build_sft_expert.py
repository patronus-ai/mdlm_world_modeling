import sys; sys.setrecursionlimit(50000)  # spacy/textworld deep import chain
"""Walk ALFWorld train games, replay the handcoded expert plan, capture full multi-turn
SFT trajectories. Each row has the system prompt, alternating user (obs+admissible) and
assistant ('Action: <cmd>') messages, ending on the winning step.
"""
import argparse
import json
import os

os.environ.setdefault("ALFWORLD_DATA", os.environ.get("ALFWORLD_DATA", "/workspace/user/alfworld/data"))

from alfworld.agents.environment import get_environment
from alfworld_prompt import AGENT_SYSTEM_PROMPT, format_user_turn

CONFIG = {
    "dataset": {
        "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
        "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
        "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
        "num_train_games": -1, "num_eval_games": -1,
    },
    "logic": {
        "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
        "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
    },
    "env": {
        "type": "AlfredTWEnv", "task_types": [1, 2, 3, 4, 5, 6],
        "expert_type": "handcoded", "goal_desc_human_anns_prob": 0.0,
        "domain_randomization": False, "hide_init_receptacles": False,
    },
    "general": {"training_method": "dagger"},
    "dagger": {"training": {"max_nb_steps_per_episode": 50}},
}


_alfred_cache = None


def _alfred():
    global _alfred_cache
    if _alfred_cache is None:
        _alfred_cache = get_environment("AlfredTWEnv")(CONFIG, train_eval="train")
    return _alfred_cache


def rollout_expert(game_file: str, max_steps: int = 50):
    alfred = _alfred()
    alfred.game_files = [game_file]
    alfred.num_games = 1
    env = alfred.init_env(batch_size=1)
    try:
        obs, info = env.reset()
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": format_user_turn(obs[0], info["admissible_commands"][0])},
        ]
        won = False
        for _ in range(max_steps):
            plan = info.get("extra.expert_plan", [[]])[0]
            if not plan:
                break
            action = plan[0]
            messages.append({"role": "assistant", "content": f"Action: {action}"})
            obs, scores, dones, info = env.step([action])
            won = bool(info["won"][0])
            if dones[0]:
                break
            messages.append({"role": "user", "content": format_user_turn(obs[0], info["admissible_commands"][0])})
        return messages, won
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1, help="parallel workers (sharded subprocesses)")
    ap.add_argument("--shard-id", type=int, default=-1, help="(internal) shard index for worker mode")
    ap.add_argument("--num-shards", type=int, default=1, help="(internal) total shards")
    args = ap.parse_args()

    # Worker mode: do the actual rollouts for assigned shard.
    if args.shard_id >= 0:
        alfred = _alfred()
        import random
        games = sorted(alfred.game_files)
        random.Random(7).shuffle(games)
        if args.limit > 0:
            games = games[:args.limit]
        my_games = games[args.shard_id::args.num_shards]
        written = 0
        with open(args.out, "w") as f:
            for i, gf in enumerate(my_games):
                try:
                    messages, won = rollout_expert(gf, args.max_steps)
                except Exception as e:
                    print(f"[shard {args.shard_id}] skip {gf}: {e}", flush=True)
                    continue
                if not won:
                    continue
                f.write(json.dumps({"messages": messages, "game_file": gf}) + "\n")
                f.flush()
                written += 1
                if (i + 1) % 25 == 0:
                    print(f"[shard {args.shard_id}] {i+1}/{len(my_games)} kept={written}", flush=True)
        print(f"[shard {args.shard_id}] wrote {written} to {args.out}", flush=True)
        return

    # Coordinator mode: spawn N worker subprocesses, each with its own tatsu parser.
    if args.workers <= 1:
        # Fall back to in-process single-shard run.
        args.shard_id, args.num_shards = 0, 1
        return main()  # re-enter as worker

    import subprocess, sys, tempfile, os as _os
    out_dir = tempfile.mkdtemp(prefix="alfworld_sft_")
    procs = []
    shard_files = []
    for s in range(args.workers):
        shard_out = _os.path.join(out_dir, f"shard_{s}.jsonl")
        shard_files.append(shard_out)
        cmd = [sys.executable, __file__,
               "--out", shard_out,
               "--limit", str(args.limit),
               "--max-steps", str(args.max_steps),
               "--shard-id", str(s),
               "--num-shards", str(args.workers)]
        log = open(_os.path.join(out_dir, f"shard_{s}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    print(f"spawned {len(procs)} workers, logs in {out_dir}", flush=True)
    for p in procs:
        p.wait()
    # Concatenate shards
    written = 0
    with open(args.out, "w") as fout:
        for sf in shard_files:
            if not _os.path.exists(sf):
                continue
            for line in open(sf):
                fout.write(line); written += 1
    print(f"wrote {written} expert trajectories to {args.out}")


if __name__ == "__main__":
    main()
