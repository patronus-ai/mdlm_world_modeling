"""
GRPO Plugin — ScienceWorld MultiTurnScheduler + reward.
v2: Major improvements based on failure analysis.

Registers:
  multi_turns['sciworld_scheduler']
  orms['sciworld_reward']

Changes from v1:
- Dynamic inventory and room tracking in scheduler
- Focus validation against room map (rejects non-existent objects)
- Deterministic responses for teleport/wait/inventory (no WM needed)
- Better loop detection (unique action diversity check)
- Increased diffusion steps (1 → 4) for WM quality
- Differentiated reward clamping (-0.5 for wrong focus vs -0.1 mild)
- Fixed short-episode penalty (amplify, not reduce)
- Action-specific WM fallbacks
"""
import json
import os
import re
import uuid
import requests
from typing import Any, Dict, List, Optional, Set

from sciworld_wm_prompt import build_wm_prompt, parse_action
from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import get_logger

logger = get_logger()

WM_ENDPOINT = os.environ.get("WM_ENDPOINT", "")
SCIWORLD_ENV_URL = os.environ.get("SCIWORLD_ENV_URL", "http://localhost:30003")

ALL_ROOMS = frozenset({
    'hallway', 'kitchen', 'workshop', 'greenhouse', 'bedroom',
    'living room', 'art studio', 'foundry', 'bathroom', 'outside',
})


def _parse_room_map(wm_system_prompt: str) -> Dict[str, str]:
    """Extract room -> contents text from the WM system prompt."""
    rooms = {}
    pattern = r'\[([^\]]+)\]\n(.*?)(?=\n\n\[|\n\n##|\Z)'
    for m in re.finditer(pattern, wm_system_prompt, re.DOTALL):
        rooms[m.group(1).strip().lower()] = m.group(2).strip().lower()
    return rooms


def _extract_initial_room(messages: list) -> str:
    """Extract starting room from the initial user message."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"is called the (.+?)\.", content)
            if m:
                return m.group(1).strip().lower()
    return "hallway"


def call_world_model(system_prompt: str, state: list, action_text: str,
                     inventory: Optional[Set[str]] = None) -> str:
    """Call the WM to predict the next observation."""
    if not WM_ENDPOINT:
        return ""

    wm_prompt = build_wm_prompt(system_prompt, state, action_text, inventory=inventory)
    payload = {
        "input": {
            "state": state,
            "action": [{"role": "assistant", "content": action_text}],
            "system_prompt": wm_prompt,
            "max_tokens": 512,
            "num_diffusion_steps": 4,
        }
    }
    try:
        if not hasattr(call_world_model, "_session"):
            call_world_model._session = requests.Session()
        resp = call_world_model._session.post(WM_ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("output", data).get("generated_text", "")
        # Strip thinking tags (matched or unmatched)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"</think>", "", raw).strip()
        raw = re.sub(r"<think>", "", raw).strip()
        raw = re.sub(r"</?tool_response>", "", raw).strip()
        raw = re.sub(r"^(?:OBSERVATION|OBS|Response)\s*:?\s*", "", raw, flags=re.IGNORECASE).strip()
        # Handle JSON-wrapped responses from Qwen3.5 WM
        if raw.startswith("[") or raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    item = parsed[0]
                    if isinstance(item, dict):
                        content = item.get("content", item.get("text", ""))
                        try:
                            inner = json.loads(content)
                            if isinstance(inner, list):
                                for part in inner:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        raw = part["text"].strip()
                                        break
                            elif isinstance(inner, dict) and "text" in inner:
                                raw = inner["text"].strip()
                            elif isinstance(inner, str):
                                raw = inner.strip()
                        except (json.JSONDecodeError, TypeError):
                            raw = str(content).strip()
                    elif isinstance(item, str):
                        raw = item.strip()
            except json.JSONDecodeError:
                pass
        return raw if raw else ""
    except Exception as e:
        logger.warning(f"WM error: {e}")
        return ""


def call_real_env(task_id: str, action: str) -> dict:
    """Call the real ScienceWorld env for ground-truth scoring."""
    try:
        r = requests.post(SCIWORLD_ENV_URL, json={"task_id": task_id, "action": action}, timeout=30)
        return r.json()
    except:
        return {"observation": "env error", "score": 0.0, "done": True}


def close_real_env(task_id: str):
    try:
        requests.delete(SCIWORLD_ENV_URL, json={"task_id": task_id}, timeout=5)
    except:
        pass


class SciWorldScheduler(MultiTurnScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wm = bool(WM_ENDPOINT)
        self._action_history = {}
        self._current_room = {}
        self._inventory = {}
        self._room_map = {}
        self._full_wm_text = {}

    def _get_req_id(self, req):
        return getattr(req, "uuid", None) or id(req)

    def _get_task(self, req):
        d = req.data_dict
        return {
            "task_id": d.get("task_id", ""),
            "instruction": d.get("instruction", ""),
            "wm_system_prompt": d.get("wm_system_prompt", ""),
            "max_turns": d.get("max_turns", 50),
        }

    def _init_state(self, req_id, infer_request):
        """Initialize tracking state for a new episode."""
        if req_id in self._room_map:
            return
        task = self._get_task(infer_request)
        wm_prompt = task.get("wm_system_prompt", "")
        self._room_map[req_id] = _parse_room_map(wm_prompt)
        self._full_wm_text[req_id] = wm_prompt.lower()
        self._inventory[req_id] = set()
        self._current_room[req_id] = _extract_initial_room(infer_request.messages)
        self._action_history[req_id] = []

    def _cleanup(self, req_id):
        """Clean up state for a finished episode."""
        self._action_history.pop(req_id, None)
        self._current_room.pop(req_id, None)
        self._inventory.pop(req_id, None)
        self._room_map.pop(req_id, None)
        self._full_wm_text.pop(req_id, None)

    def _object_exists_in_env(self, req_id, target: str) -> bool:
        """Check if an object exists anywhere in the environment."""
        target_lower = target.lower()
        if target_lower in self._full_wm_text.get(req_id, ""):
            return True
        if target_lower in self._inventory.get(req_id, set()):
            return True
        return False

    def check_finished(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        action = parse_action(completion)

        if not action:
            return True

        req_id = self._get_req_id(infer_request)
        self._init_state(req_id, infer_request)

        history = self._action_history.get(req_id, [])
        history.append(action)
        self._action_history[req_id] = history

        # AAA: 3 identical in a row
        if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
            self._cleanup(req_id)
            return True
        # ABAB: oscillation
        if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4] and history[-1] != history[-2]:
            self._cleanup(req_id)
            return True
        # Low action diversity: < 5 unique actions in last 20 turns
        if len(history) >= 20:
            recent = history[-20:]
            if len(set(recent)) < 5:
                self._cleanup(req_id)
                return True

        # Context length check
        n_msgs = len(infer_request.messages)
        if n_msgs > 80:
            self._cleanup(req_id)
            return True

        result = super().check_finished(infer_request, response_choice, current_turn)
        if result:
            self._cleanup(req_id)
        return result

    def step(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        task = self._get_task(infer_request)
        action = parse_action(completion)

        if not action:
            infer_request.messages.append({
                "role": "user",
                "content": "Please provide a valid action. Examples: 'look around', 'teleport to kitchen', 'pick up thermometer'.",
            })
            return {"infer_request": infer_request}

        req_id = self._get_req_id(infer_request)
        self._init_state(req_id, infer_request)

        action_lower = action.lower().strip()

        # --- Deterministic responses (no WM call needed) ---

        # Teleport always succeeds
        tp = re.match(r"teleport to (.+)", action_lower)
        if tp:
            dest = tp.group(1).strip()
            self._current_room[req_id] = dest
            infer_request.messages.append({
                "role": "user",
                "content": f"You teleport to the {dest}.",
            })
            return {"infer_request": infer_request}

        # Wait always succeeds
        if action_lower in ("wait", "wait1"):
            infer_request.messages.append({
                "role": "user",
                "content": "(1 tick passes)",
            })
            return {"infer_request": infer_request}

        # Inventory — list tracked items
        if action_lower == "inventory":
            inv = self._inventory.get(req_id, set())
            if inv:
                obs = f"In your inventory, you see: {', '.join(sorted(inv))}"
            else:
                obs = "Your inventory is empty."
            infer_request.messages.append({"role": "user", "content": obs})
            return {"infer_request": infer_request}

        # --- Focus validation ---
        fm = re.match(r"focus on (.+)", action_lower)
        if fm:
            target = fm.group(1).strip()
            if target == "agent" or not self._object_exists_in_env(req_id, target):
                infer_request.messages.append({
                    "role": "user",
                    "content": "No known action matches that input.",
                })
                return {"infer_request": infer_request}

        # --- WM call for all other actions ---
        observation = ""
        if self.use_wm:
            wm_prompt = task.get("wm_system_prompt", "")
            inv = self._inventory.get(req_id, set())
            observation = call_world_model(wm_prompt, infer_request.messages, action, inventory=inv)

        # Action-specific fallbacks when WM returns empty
        if not observation:
            if action_lower.startswith("look around"):
                room = self._current_room.get(req_id, "")
                room_map = self._room_map.get(req_id, {})
                room_text = room_map.get(room, "")
                if room_text:
                    observation = room_text
                else:
                    observation = f"This room is called the {room}. You look around but see nothing notable."
            elif action_lower.startswith("go to"):
                gm = re.match(r"go to (.+)", action_lower)
                if gm:
                    observation = f"You move to the {gm.group(1).strip()}."
                else:
                    observation = "No known action matches that input."
            else:
                observation = "No known action matches that input."

        # --- Track state changes from the observation ---
        obs_lower = observation.lower()

        # Track pick up
        pm = re.match(r"pick up (.+)", action_lower)
        if pm and "you move the" in obs_lower and "inventory" in obs_lower:
            item = pm.group(1).strip()
            self._inventory.setdefault(req_id, set()).add(item)

        # Track room change from "go to"
        gm = re.match(r"go to (.+)", action_lower)
        if gm and ("you move to" in obs_lower or "you go to" in obs_lower):
            self._current_room[req_id] = gm.group(1).strip()

        # Track put (remove from inventory)
        ptm = re.match(r"put (.+?) (?:in|on) (.+)", action_lower)
        if ptm and "you move the" in obs_lower:
            self._inventory.get(req_id, set()).discard(ptm.group(1).strip())

        infer_request.messages.append({
            "role": "user",
            "content": observation,
        })
        return {"infer_request": infer_request}


multi_turns["sciworld_scheduler"] = SciWorldScheduler


class SciWorldReward(ORM):
    """Reward based on replaying actions against the real ScienceWorld environment.

    Each completion gets a unique env instance to avoid state corruption.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        messages_list = kwargs.get("messages", [[] for _ in completions])
        task_ids = kwargs.get("task_id", [""] * len(completions))

        rewards = []
        for i, completion in enumerate(completions):
            messages = messages_list[i] if i < len(messages_list) else []
            base_task_id = task_ids[i] if i < len(task_ids) else ""

            all_actions = []
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    action = parse_action(m.get("content", ""))
                    if action:
                        all_actions.append(action)
            final_action = parse_action(completion)
            if final_action:
                all_actions.append(final_action)

            if not all_actions or not base_task_id:
                rewards.append(0.0)
                continue

            unique_id = f"{base_task_id}_gen{uuid.uuid4().hex[:6]}"

            score = 0.0
            for action in all_actions:
                resp = call_real_env(unique_id, action)
                score = resp.get("score", 0.0)
                if resp.get("done", False):
                    break
            close_real_env(unique_id)

            reward = score / 100.0

            # Differentiated clamping: wrong focus vs mild negative
            if reward < 0:
                if score <= -50:
                    reward = -0.2
                else:
                    reward = max(reward, -0.1)

            # No-submission penalty: agent explored but never focused
            has_focus = any("focus on" in a.lower() for a in all_actions)
            if not has_focus and len(all_actions) >= 10:
                reward = min(reward, -0.15)

            # Short bad episodes: amplify penalty (not reduce)
            if len(all_actions) <= 2 and reward < 0:
                reward = min(reward * 1.5, -0.1)

            # Loop penalty
            n_repeated = 0
            for j in range(2, len(all_actions)):
                if all_actions[j] == all_actions[j-1] == all_actions[j-2]:
                    n_repeated += 1
            if len(all_actions) >= 4:
                for j in range(3, len(all_actions)):
                    if all_actions[j] == all_actions[j-2] and all_actions[j-1] == all_actions[j-3] and all_actions[j] != all_actions[j-1]:
                        n_repeated += 1
            if n_repeated > 0:
                reward *= max(0.3, 1.0 - 0.15 * n_repeated)

            rewards.append(max(min(reward, 1.0), -0.2))

        return rewards


orms["sciworld_reward"] = SciWorldReward
