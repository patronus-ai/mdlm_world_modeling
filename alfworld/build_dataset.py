import sys; sys.setrecursionlimit(50000)  # spacy/textworld import chain needs deep stack
"""Walk ALFWorld game files, render initial obs + admissible commands, AND bake the
WM system prompt + goal metadata into each row so GRPO can run without the real env.

Each row:
  messages: [{system}, {user with init_obs+admissible}]
  game_file, split, task_type, target_obj, destination, mrecep, toggle_target
  wm_system_prompt: per-task prompt (initial scene, receptacles, contents, goal predicates)
  init_admissible, expert_plan
"""
import argparse
import json
import os
import re

os.environ.setdefault("ALFWORLD_DATA", os.environ.get("ALFWORLD_DATA", "/workspace/user/alfworld/data"))

from alfworld.agents.environment import get_environment
from alfworld_prompt import AGENT_SYSTEM_PROMPT, format_user_turn
from alfworld_wm_prompt import parse_task_metadata

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


_RECEP_RE = re.compile(r"a ([\w ]+\d+)")


def _parse_initial_view(obs0: str) -> list[str]:
    """Extract the receptacle list from 'Looking quickly around you, you see X, Y, ...'."""
    m = re.search(r"you see (.+?)\.", obs0, re.DOTALL)
    if not m:
        return []
    return [r.strip() for r in re.findall(r"a ([\w ]+\d+)", m.group(0))]


_GO_OPEN     = re.compile(r"You arrive at ([\w ]+\d+)\. On the \1, you see (.+?)\.", re.S)
_GO_NOTHING  = re.compile(r"You arrive at ([\w ]+\d+)\. On the \1, you see nothing\.")
_GO_CLOSED   = re.compile(r"You arrive at ([\w ]+\d+)\. The \1 is closed\.")
_OPEN_CONT   = re.compile(r"You open the ([\w ]+\d+)\. The \1 is open\. In it, you see (.+?)\.", re.S)
_OPEN_EMPTY  = re.compile(r"You open the ([\w ]+\d+)\. The \1 is open\. In it, you see nothing\.")


def _split_listed(items: str) -> list[str]:
    items = items.strip()
    if items.lower() == "nothing":
        return []
    chunks = re.split(r",\s*(?:and )?| and ", items)
    out = []
    for c in chunks:
        c = c.strip()
        c = re.sub(r"^a |^an ", "", c)
        if c and c.lower() != "nothing":
            out.append(c)
    return out


def walk_env_for_scene(env, receptacles: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Walk the env to record per-receptacle (state, contents).

    For each receptacle:
      - go to it (one step)
      - if response says 'closed' -> open it -> record contents -> close it
      - else parse contents from the 'go to' response

    Returns (recep_state, recep_contents):
      recep_state[recep] in {'open', 'closed'}
      recep_contents[recep] = ['mug 1', 'pot 1', ...]   (empty list if nothing)
    """
    recep_state: dict[str, str] = {}
    recep_contents: dict[str, list[str]] = {}

    for recep in receptacles:
        # 1. go to recep
        try:
            obs, _, dones, _ = env.step([f"go to {recep}"])
        except Exception:
            continue
        text = obs[0]
        if dones[0]:
            break

        m = _GO_CLOSED.search(text)
        if m and m.group(1) == recep:
            # 2. open it
            obs, _, dones, _ = env.step([f"open {recep}"])
            text2 = obs[0]
            # Check empty FIRST so "you see nothing." doesn't get matched by _OPEN_CONT.
            if _OPEN_EMPTY.search(text2):
                recep_state[recep] = "closed"
                recep_contents[recep] = []
            else:
                mo = _OPEN_CONT.search(text2)
                if mo and mo.group(1) == recep:
                    recep_state[recep] = "closed"  # *initial* state was closed
                    recep_contents[recep] = _split_listed(mo.group(2))
                else:
                    # 'open' failed unexpectedly; treat as not-openable / leave as closed unknown
                    recep_state[recep] = "closed"
                    recep_contents[recep] = []
            # 3. close it back so the agent's first 'go to' sees the same closed state
            try:
                env.step([f"close {recep}"])
            except Exception:
                pass
            continue

        m_open = _GO_OPEN.search(text)
        m_nothing = _GO_NOTHING.search(text)
        if m_open and m_open.group(1) == recep:
            recep_state[recep] = "open"
            recep_contents[recep] = _split_listed(m_open.group(2))
        elif m_nothing and m_nothing.group(1) == recep:
            recep_state[recep] = "open"
            recep_contents[recep] = []
        # else: unrecognised — skip

    return recep_state, recep_contents


def build_wm_system_prompt(traj_data: dict, init_obs: str,
                           recep_state: dict[str, str], recep_contents: dict[str, list[str]]) -> str:
    """Bake a per-task WM system prompt with full scene knowledge.

    Sections:
      Task: <goal text>
      Task type: <task_type>
      Goal predicates: target=..., destination=..., needs_clean=..., toggle=...
      ## INITIAL SCENE
        - receptacle 1 (open|closed)
        - receptacle 2 (open|closed)
        ...
      ## INITIAL OBJECT LOCATIONS
        - obj 1: in/on receptacle X
        - obj 2: in/on receptacle Y
        ...
    """
    task_type = traj_data.get("task_type", "")
    pddl = traj_data.get("pddl_params", {}) or {}
    meta = parse_task_metadata(task_type, pddl)

    goal = init_obs.split("Your task is to:", 1)[-1].strip().split("\n", 1)[0].strip()
    receptacles = _parse_initial_view(init_obs)

    parts: list[str] = []
    parts.append(f"Task: {goal}")
    parts.append(f"Task type: {task_type}")
    pred_bits = [f"target={meta['target_obj']}", f"destination={meta['destination']}"]
    if meta["mrecep"]: pred_bits.append(f"mrecep={meta['mrecep']}")
    if meta["toggle_target"]: pred_bits.append(f"toggle={meta['toggle_target']}")
    if meta["needs_clean"]:  pred_bits.append("needs_clean=true")
    if meta["needs_heat"]:   pred_bits.append("needs_heat=true")
    if meta["needs_cool"]:   pred_bits.append("needs_cool=true")
    if meta["needs_two"]:    pred_bits.append("needs_two=true")
    if meta["needs_toggle"]: pred_bits.append("needs_toggle=true")
    parts.append("Goal predicates: " + ", ".join(pred_bits))
    parts.append("")

    parts.append("## INITIAL SCENE")
    for r in receptacles:
        st = recep_state.get(r, "open")  # default to open if walk skipped it
        parts.append(f"- {r} ({st})")
    parts.append("")

    parts.append("## INITIAL OBJECT LOCATIONS")
    any_listed = False
    for r in receptacles:
        items = recep_contents.get(r, [])
        for obj in items:
            if not obj or obj.lower() == "nothing":
                continue
            # Use 'in' for closed-default classes, 'on' otherwise
            preposition = "in" if recep_state.get(r) == "closed" else "on"
            parts.append(f"- {obj}: {preposition} {r}")
            any_listed = True
    if not any_listed:
        parts.append("(no objects observed in any visited receptacle)")

    return "\n".join(parts)


_alfred_cache: dict[str, object] = {}


def _alfred(split: str):
    if split not in _alfred_cache:
        _alfred_cache[split] = get_environment("AlfredTWEnv")(CONFIG, train_eval=split)
    return _alfred_cache[split]


def render_initial(game_file: str, split: str):
    """Reset env, walk receptacles, return (init_obs, admissible, expert_plan, recep_state, recep_contents)."""
    alfred = _alfred(split)
    alfred.game_files = [game_file]
    alfred.num_games = 1
    env = alfred.init_env(batch_size=1)
    try:
        obs, info = env.reset()
        init_text = obs[0]
        admissible = info["admissible_commands"][0]
        expert_plan = info.get("extra.expert_plan", [[]])[0]
        receptacles = _parse_initial_view(init_text)
        recep_state, recep_contents = walk_env_for_scene(env, receptacles)
    finally:
        env.close()
    return init_text, admissible, expert_plan, recep_state, recep_contents


def main():
    import sys as _sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True,
                    choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle games (for balanced task-type coverage when --limit is set)")
    args = ap.parse_args()

    alfred = _alfred(args.split)
    games = sorted(alfred.game_files)
    if args.shuffle:
        import random
        random.Random(7).shuffle(games)
    if args.limit > 0:
        games = games[:args.limit]
    print(f"Processing {len(games)} games for split={args.split}", flush=True)
    _run_loop(games, args.split, args.out)


def _run_loop(games, split, out_path, prefix=""):
    written = 0
    with open(out_path, "w") as f:
        for i, gf in enumerate(games):
            try:
                init_obs, admissible, expert_plan, recep_state, recep_contents = render_initial(gf, split)
            except Exception as e:
                print(f"{prefix}skip {gf}: {e}", flush=True)
                continue
            traj_path = os.path.join(os.path.dirname(gf), "traj_data.json")
            try:
                traj = json.load(open(traj_path))
            except Exception as e:
                print(f"{prefix}no traj_data for {gf}: {e}", flush=True)
                continue
            meta = parse_task_metadata(traj.get("task_type", ""), traj.get("pddl_params", {}) or {})
            wm_sys = build_wm_system_prompt(traj, init_obs, recep_state, recep_contents)
            user = format_user_turn(init_obs, admissible)
            row = {
                "messages": [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "game_file": gf,
                "split": split,
                "task_type":     meta["task_type"],
                "target_obj":    meta["target_obj"],
                "destination":   meta["destination"],
                "mrecep":        meta["mrecep"],
                "toggle_target": meta["toggle_target"],
                "wm_system_prompt": wm_sys,
                "init_admissible": admissible,
                "expert_plan":   expert_plan,
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            written += 1
            if (i + 1) % 25 == 0:
                print(f"{prefix}{i+1}/{len(games)} kept={written}", flush=True)
    print(f"{prefix}wrote {written} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
