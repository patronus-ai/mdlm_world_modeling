#!/usr/bin/env python3
import sys; sys.setrecursionlimit(50000)  # spacy/textworld deep import chain
"""HTTP server wrapping AlfredTWEnv for training + eval.

POST /reset  {task_id, game_file?, split?}  -> {obs, admissible, goal, gamefile}
POST /step   {task_id, action}              -> {obs, admissible, won, done, score}
DELETE /close {task_id}                     -> {status}
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("ALFWORLD_DATA", os.environ.get("ALFWORLD_DATA", "/workspace/user/alfworld/data"))

from alfworld.agents.environment import get_environment


CONFIG = {
    "dataset": {
        "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
        "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
        "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
        "num_train_games": -1,
        "num_eval_games": -1,
    },
    "logic": {
        "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
        "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
    },
    "env": {
        "type": "AlfredTWEnv",
        "task_types": [1, 2, 3, 4, 5, 6],
        "expert_type": "handcoded",
        "goal_desc_human_anns_prob": 0.0,
        "domain_randomization": False,
        "hide_init_receptacles": False,
    },
    "general": {"training_method": "dagger"},
    "dagger": {"training": {"max_nb_steps_per_episode": 50}},
}

_lock = threading.Lock()
_envs = {}        # task_id -> textworld gym env
_meta = {}        # task_id -> {goal, gamefile}
_alfreds = {}     # split -> AlfredTWEnv (collects game files)


def _alfred(split: str):
    if split not in _alfreds:
        _alfreds[split] = get_environment("AlfredTWEnv")(CONFIG, train_eval=split)
    return _alfreds[split]


def _make_env(game_file: str | None, split: str):
    alfred = _alfred(split)
    if game_file:
        alfred.game_files = [game_file]
        alfred.num_games = 1
    env = alfred.init_env(batch_size=1)
    return env


def _extract_goal(obs: str) -> str:
    if "Your task is to:" in obs:
        return obs.split("Your task is to:", 1)[1].strip()
    return ""


class Handler(BaseHTTPRequestHandler):
    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _ok(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self._read()
        task_id = body.get("task_id")
        if not task_id:
            return self._ok({"error": "task_id required"})

        # textworld's tatsu parser holds module-global state, so all env ops
        # (reset+step) must be serialised. OpenAI/network calls on the client
        # side still parallelise.
        if self.path == "/reset":
            game_file = body.get("game_file")
            split = body.get("split", "train")
            with _lock:
                if task_id in _envs:
                    try: _envs[task_id].close()
                    except: pass
                env = _make_env(game_file, split)
                _envs[task_id] = env
                obs, info = env.reset()
                text = obs[0]
                admissible = info["admissible_commands"][0]
                gamefile = info.get("extra.gamefile", [""])[0]
            goal = _extract_goal(text)
            _meta[task_id] = {"goal": goal, "gamefile": gamefile}
            return self._ok({"obs": text, "admissible": admissible,
                             "goal": goal, "gamefile": gamefile})

        if self.path == "/step":
            action = body.get("action", "")
            env = _envs.get(task_id)
            if env is None:
                return self._ok({"error": "no env for task_id; call /reset first"})
            with _lock:
                obs, scores, dones, info = env.step([action])
                payload = {
                    "obs": obs[0],
                    "admissible": info["admissible_commands"][0],
                    "won": bool(info["won"][0]),
                    "done": bool(dones[0]),
                    "score": float(scores[0]),
                }
            return self._ok(payload)

        return self._ok({"error": f"unknown route {self.path}"})

    def do_DELETE(self):
        body = self._read()
        task_id = body.get("task_id")
        with _lock:
            env = _envs.pop(task_id, None)
            _meta.pop(task_id, None)
        if env is not None:
            try: env.close()
            except: pass
        return self._ok({"status": "closed"})

    def log_message(self, *a, **kw): pass


if __name__ == "__main__":
    port = int(os.environ.get("ALFWORLD_PORT", "30003"))
    print(f"AlfredTWEnv server on port {port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
