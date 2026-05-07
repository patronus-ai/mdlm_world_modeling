"""GRPO plugin — ALFWorld MultiTurnScheduler + reward.

Training loop (no real env): action -> deterministic local responder -> if it
abstains, fall back to the SDAR WM at WM_ENDPOINT. Reward is derived from the
trajectory by checking ALFWorld goal predicates (no env access at training).

The real AlfredTWEnv is reserved for evaluation only (see eval.py).

Registers:
  multi_turns['alfworld_scheduler']
  orms['alfworld_reward']
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

import requests

from alfworld_prompt import format_user_turn
from alfworld_wm_prompt import (
    admissible_commands,
    build_alfworld_wm_prompt,
    check_goal_satisfied,
    expected_alfworld_response,
    replay_state,
    NOTHING_HAPPENS,
)
from swift.infer_engine.protocol import RolloutInferRequest
from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import get_logger

logger = get_logger()

# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

WM_ENDPOINT       = os.environ.get("WM_ENDPOINT", "http://localhost:30000")
RUNPOD_API_KEY    = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
TRAJECTORY_LOG    = os.environ.get("TRAJECTORY_LOG", "/tmp/alfworld_trajectories.jsonl")
WM_GUARD          = os.environ.get("ALFWORLD_WM_GUARD", "1") == "1"  # use local responder when possible

ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# Safeguards. Override via env vars; mirrored in eval.py.
LOOP_REPEAT_LIMIT     = int(os.environ.get("ALFWORLD_LOOP_LIMIT", "3"))   # consecutive identical actions -> kill
NOTHING_STREAK_LIMIT  = int(os.environ.get("ALFWORLD_NOTHING_LIMIT", "4")) # consecutive 'Nothing happens.' -> kill

_wm_semaphore = threading.Semaphore(int(os.environ.get("WM_MAX_CONCURRENT", "8")))


def parse_action(text: str) -> Optional[str]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = ACTION_RE.search(text)
    if m:
        action = m.group(1).strip().rstrip(".").strip()
    else:
        action = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not action:
        return None
    # Normalize case — ALFWorld is lowercase
    action = action[0].lower() + action[1:] if action else action
    # Strip leading "the " / "a " from articles agents sometimes insert
    action = re.sub(r"\b(?:the|a|an) ([\w]+ \d+)", r"\1", action, flags=re.I)
    return action


# --------------------------------------------------------------------------------
# WM client
# --------------------------------------------------------------------------------

def _unwrap_wm_response(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?tool_response>", "", text).strip()
    text = text.strip().strip("`").strip()
    # Handle JSON-wrapped responses from Qwen3.5 WM
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                item = parsed[0]
                if isinstance(item, dict) and "content" in item:
                    content = item["content"]
                    try:
                        inner = json.loads(content)
                        if isinstance(inner, list):
                            for part in inner:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    return part["text"].strip() or NOTHING_HAPPENS
                        elif isinstance(inner, str):
                            return inner.strip() or NOTHING_HAPPENS
                    except (json.JSONDecodeError, TypeError):
                        return str(content).strip() or NOTHING_HAPPENS
                elif isinstance(item, str):
                    return item.strip() or NOTHING_HAPPENS
        except json.JSONDecodeError:
            pass
    return text or NOTHING_HAPPENS


def call_world_model(system_prompt: str, state: list, action: str, max_tokens: int = 256) -> str:
    """Predict next observation. Tries RunPod first, then local WM_ENDPOINT."""
    import time as _t

    payload_input = {
        "system_prompt": system_prompt,
        "state": state,
        "action": action,
        "max_tokens": max_tokens,
        "num_diffusion_steps": 1,
    }

    if RUNPOD_ENDPOINT_ID:
        _wm_semaphore.acquire()
        try:
            base = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
            headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
            r = requests.post(f"{base}/run", json={"input": payload_input}, headers=headers, timeout=30)
            r.raise_for_status()
            job_id = r.json().get("id")
            if not job_id:
                return NOTHING_HAPPENS
            for _ in range(15):
                _t.sleep(2)
                s = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=10).json()
                if s.get("status") == "COMPLETED":
                    out = s.get("output", {})
                    raw = out.get("generated_text", out.get("text", "")) if isinstance(out, dict) else str(out)
                    return _unwrap_wm_response(raw)
                if s.get("status") == "FAILED":
                    return NOTHING_HAPPENS
            return NOTHING_HAPPENS
        except Exception as e:
            logger.warning(f"WM runpod error: {e}")
            return NOTHING_HAPPENS
        finally:
            _wm_semaphore.release()

    if not WM_ENDPOINT:
        return NOTHING_HAPPENS
    if not hasattr(call_world_model, "_session"):
        call_world_model._session = requests.Session()
    try:
        r = call_world_model._session.post(WM_ENDPOINT, json={"input": payload_input}, timeout=120)
        r.raise_for_status()
        data = r.json()
        raw = data.get("output", data).get("generated_text", "") if isinstance(data.get("output", data), dict) else ""
        return _unwrap_wm_response(raw)
    except Exception as e:
        logger.warning(f"WM endpoint error: {e}")
        return NOTHING_HAPPENS


# --------------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------------

class AlfworldScheduler(MultiTurnScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dones: set = set()
        self._meta:  dict = {}   # req_id -> goal_meta dict (parsed from data_dict)
        self._actions: dict = {} # req_id -> list[str] of recent actions (for loop kill)
        self._nothing: dict = {} # req_id -> int (consecutive 'Nothing happens.' streak)
        self._kills:   dict = {} # req_id -> reason string (for reward / log)

    def _req_id(self, infer_request) -> str:
        return getattr(infer_request, "uuid", None) or str(id(infer_request))

    def _ensure_meta(self, infer_request):
        rid = self._req_id(infer_request)
        if rid in self._meta:
            return self._meta[rid]
        d = infer_request.data_dict or {}
        meta = {
            "task_type":     d.get("task_type", ""),
            "target_obj":    (d.get("target_obj")    or "").lower(),
            "destination":   (d.get("destination")   or "").lower(),
            "mrecep":        (d.get("mrecep")        or "").lower(),
            "toggle_target": (d.get("toggle_target") or "").lower(),
            "wm_system_prompt": d.get("wm_system_prompt", ""),
        }
        self._meta[rid] = meta
        return meta

    def check_finished(self, infer_request, response_choice, current_turn):
        rid = self._req_id(infer_request)
        if rid in self._dones:
            return True
        # early-stop on satisfied goal
        meta = self._ensure_meta(infer_request)
        if check_goal_satisfied(meta, infer_request.messages):
            self._dones.add(rid)
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn):
        rid = self._req_id(infer_request)
        meta = self._ensure_meta(infer_request)
        completion = response_choice.message.content or ""
        action = parse_action(completion)

        if not action:
            infer_request.messages.append({
                "role": "user",
                "content": "Your last response was not parseable. Reply with exactly one line: 'Action: <command>'.",
            })
            self._dones.add(rid); self._kills[rid] = "unparseable"
            return {"infer_request": infer_request}

        # Loop-kill: AAA (3 identical) or ABAB (oscillation between 2 actions)
        hist = self._actions.setdefault(rid, [])
        hist.append(action.lower())
        is_loop = False
        if len(hist) >= LOOP_REPEAT_LIMIT and len(set(hist[-LOOP_REPEAT_LIMIT:])) == 1:
            is_loop = True
        if len(hist) >= 4 and hist[-1] == hist[-3] and hist[-2] == hist[-4] and hist[-1] != hist[-2]:
            is_loop = True
        if len(hist) >= 20 and len(set(hist[-20:])) <= 3:
            is_loop = True  # low diversity over 20 turns
        if is_loop:
            infer_request.messages.append({
                "role": "user",
                "content": f"<tool_response>\n[episode terminated by scheduler: action {action!r} repeated {LOOP_REPEAT_LIMIT} times]\n</tool_response>",
            })
            self._dones.add(rid); self._kills[rid] = "loop"
            return {"infer_request": infer_request}

        wm_prompt = meta["wm_system_prompt"]

        # 1. deterministic local responder
        local = expected_alfworld_response(wm_prompt, infer_request.messages, action) if WM_GUARD else None

        # 2. fall back to the SDAR WM
        if local is None:
            obs_text = call_world_model(
                build_alfworld_wm_prompt(wm_prompt, infer_request.messages, action),
                infer_request.messages,
                action,
            )
        else:
            obs_text = local

        # Append to conversation. Recompute admissible commands from the post-step state
        # so every user turn matches the SFT format (Observation + Admissible commands).
        new_state = replay_state(wm_prompt, infer_request.messages + [
            {"role": "user", "content": f"<tool_response>\n{obs_text}\n</tool_response>"}
        ])
        adm = admissible_commands(new_state)
        infer_request.messages.append({
            "role": "user",
            "content": format_user_turn(obs_text, adm),
        })

        # Track 'Nothing happens.' streak — kill if NOTHING_STREAK_LIMIT consecutive.
        if obs_text.strip() == NOTHING_HAPPENS:
            self._nothing[rid] = self._nothing.get(rid, 0) + 1
        else:
            self._nothing[rid] = 0
        if self._nothing.get(rid, 0) >= NOTHING_STREAK_LIMIT:
            infer_request.messages.append({
                "role": "user",
                "content": f"<tool_response>\n[episode terminated by scheduler: {NOTHING_STREAK_LIMIT} consecutive invalid actions]\n</tool_response>",
            })
            self._dones.add(rid); self._kills[rid] = "stuck"
            return {"infer_request": infer_request}

        # check goal & log
        won = check_goal_satisfied(meta, infer_request.messages)

        try:
            with open(TRAJECTORY_LOG, "a") as f:
                f.write(json.dumps({
                    "rid": rid, "turn": current_turn, "action": action,
                    "obs": obs_text[:240], "won": won,
                    "via": "local" if local is not None else "wm",
                }) + "\n")
        except Exception:
            pass

        if won:
            self._dones.add(rid)
        return {"infer_request": infer_request}


multi_turns["alfworld_scheduler"] = AlfworldScheduler


# --------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------

class AlfworldReward(ORM):
    """Reward = 1.0 if goal predicates satisfied, else partial credit.

    Partial credit:
      + 0.10 took the target object instance (for any task type)
      + 0.10 prerequisite achieved (clean/heat/cool/use depending on task type)
      + 0.05 placed *something* in destination receptacle
      + small format multiplier penalising malformed actions
    """

    def _format_multiplier(self, completion: str, action: Optional[str]) -> float:
        if not action:
            return 0.3
        # Penalise prose preamble before 'Action:'
        text = re.sub(r"<think>.*?</think>", "", completion or "", flags=re.DOTALL)
        m = ACTION_RE.search(text)
        if m and len(text[:m.start()].strip()) > 32:
            return 0.7
        return 1.0

    def _trajectory_quality(self, full: List[dict]) -> Dict[str, float]:
        """Count repeats and 'Nothing happens.' from the assistant action history."""
        actions = []
        nothing_streak_max = 0
        nothing_streak = 0
        n_actions = 0
        n_nothing = 0
        for m in full:
            if not isinstance(m, dict):
                continue
            content = str(m.get("content") or "")
            role = m.get("role")
            if role == "assistant":
                a = parse_action(content)
                if a:
                    actions.append(a.lower())
                    n_actions += 1
            elif role == "user":
                # match the verbatim "Nothing happens." lifted into the tool_response wrap
                if "Nothing happens." in content:
                    nothing_streak += 1
                    n_nothing += 1
                    nothing_streak_max = max(nothing_streak_max, nothing_streak)
                else:
                    nothing_streak = 0
        # longest run of identical consecutive actions
        max_repeat = 1
        run = 1
        for i in range(1, len(actions)):
            if actions[i] == actions[i - 1]:
                run += 1
                max_repeat = max(max_repeat, run)
            else:
                run = 1
        return {
            "n_actions": n_actions,
            "n_nothing": n_nothing,
            "max_repeat": max_repeat,
            "max_nothing_streak": nothing_streak_max,
        }

    def __call__(self, completions, **kwargs) -> List[float]:
        messages_list   = kwargs.get("messages", [[] for _ in completions])
        target_objs     = kwargs.get("target_obj", [""] * len(completions))
        destinations    = kwargs.get("destination", [""] * len(completions))
        task_types      = kwargs.get("task_type", [""] * len(completions))
        toggle_targets  = kwargs.get("toggle_target", [""] * len(completions))
        mreceps         = kwargs.get("mrecep", [""] * len(completions))

        rewards: List[float] = []
        for i, completion in enumerate(completions):
            messages = messages_list[i] if i < len(messages_list) else []
            full = list(messages) + [{"role": "assistant", "content": completion or ""}]

            meta = {
                "target_obj":    (target_objs[i]    or "").lower() if i < len(target_objs) else "",
                "destination":   (destinations[i]   or "").lower() if i < len(destinations) else "",
                "task_type":      task_types[i]                    if i < len(task_types) else "",
                "toggle_target": (toggle_targets[i] or "").lower() if i < len(toggle_targets) else "",
                "mrecep":        (mreceps[i]        or "").lower() if i < len(mreceps) else "",
            }

            won = check_goal_satisfied(meta, full)
            qual = self._trajectory_quality(full)
            # Penalty multiplier from loop / nothing-happens behaviour. Stays in [0.4, 1.0]
            # so winning episodes still get most of the reward.
            n_act = max(qual["n_actions"], 1)
            nothing_frac = qual["n_nothing"] / n_act
            penalty = 1.0
            if qual["max_repeat"] >= LOOP_REPEAT_LIMIT:
                penalty *= 0.7
            if qual["max_nothing_streak"] >= NOTHING_STREAK_LIMIT:
                penalty *= 0.6
            penalty *= max(0.5, 1.0 - nothing_frac)

            if won:
                rewards.append(min(1.0, 1.0 * penalty))
                continue

            # partial credit
            full_text = "\n".join(str(m.get("content", "")) for m in full if isinstance(m, dict))
            target = meta["target_obj"]
            dest = meta["destination"]

            r = 0.0
            did_take = bool(target and re.search(rf"You pick up the {re.escape(target)}\b", full_text))
            did_place = bool(dest and re.search(rf"You move the [\w ]+ to the {re.escape(dest)}\b", full_text))
            did_transform = False
            if meta.get("task_type") == "pick_clean_then_place_in_recep" and re.search(rf"You clean the {re.escape(target)}\b", full_text):
                did_transform = True
            if meta.get("task_type") == "pick_heat_then_place_in_recep"  and re.search(rf"You heat the {re.escape(target)}\b",  full_text):
                did_transform = True
            if meta.get("task_type") == "pick_cool_then_place_in_recep"  and re.search(rf"You cool the {re.escape(target)}\b",  full_text):
                did_transform = True
            did_toggle = False
            if meta.get("task_type") == "look_at_obj_in_light" and meta.get("toggle_target"):
                if re.search(rf"You turn on the {re.escape(meta['toggle_target'])}\b", full_text):
                    did_toggle = True

            # Partial credit only for COMPLETED steps in the pipeline
            # take alone = 0 (no reward for hoarding)
            # take + place = 0.25
            # take + transform + place = 0.35
            # take + toggle (look_at_obj) = 0.25
            if did_place:
                r = 0.25
                if did_transform:
                    r = 0.35
            elif did_toggle and did_take:
                r = 0.25
            elif did_take and not did_place:
                r = 0.03  # small credit to keep take behavior alive

            # Penalty for explore-only or take-only episodes
            if r == 0.0 and qual["n_actions"] >= 10:
                r = -0.05

            action = parse_action(completion)
            r *= self._format_multiplier(completion, action)
            r *= penalty
            rewards.append(min(r, 0.9))
        return rewards


orms["alfworld_reward"] = AlfworldReward
