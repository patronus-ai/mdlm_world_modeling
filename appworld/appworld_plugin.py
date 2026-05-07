"""
GRPO Plugin — AppWorld MultiTurnScheduler + reward.

Registers:
  multi_turns['appworld_scheduler']
  orms['appworld_reward']
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from appworld_prompt import CANONICAL_TOOL_FORMAT
from appworld_wm_prompt import build_action_local_wm_prompt, expected_appworld_response
from fix_tool_names_and_schemas import TOOL_DEFS
from swift.infer_engine.protocol import ChatCompletionResponseChoice, RolloutInferRequest
from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WM_ENDPOINT = os.environ.get("WM_ENDPOINT", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
TRAJECTORY_LOG = os.environ.get("TRAJECTORY_LOG", "/tmp/appworld_trajectories.jsonl")

# Rate limiting for RunPod WM calls
import threading
_wm_semaphore = threading.Semaphore(int(os.environ.get("WM_MAX_CONCURRENT", "8")))

# APIs that don't count toward reward
EXCLUDED_APIS = {"login", "show_account_passwords", "complete_task"}
VALID_TOOL_NAMES = {tool["name"] for tool in TOOL_DEFS}


# ---------------------------------------------------------------------------
# WM client
# ---------------------------------------------------------------------------

def _unwrap_wm_response(raw: str) -> str:
    text = raw.strip()
    # Strip thinking and tool_response tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'</?tool_response>', '', text).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # WM often drops commas between JSON objects — use json_repair
        try:
            import json_repair
            parsed = json_repair.loads(text)
            text = json.dumps(parsed)
        except Exception:
            return text
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if isinstance(parsed, dict) and "content" in parsed:
        return str(parsed["content"])
    return text


def call_world_model(system_prompt: str, state: list, action: list) -> str:
    import time as _time

    # RunPod serverless endpoint (rate limited)
    if RUNPOD_ENDPOINT_ID:
        _wm_semaphore.acquire()
        base_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {
                "system_prompt": system_prompt,
                "state": state,
                "action": action if action else [],
                "max_tokens": 2048,
                "num_diffusion_steps": 1,
            }
        }
        try:
            # Use /run (async) + polling to avoid runsync timeout
            resp = requests.post(f"{base_url}/run", json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            job_id = data.get("id")

            if not job_id:
                return json.dumps({"error": "no job_id"})

            # Poll for result
            for _ in range(15):  # Up to 30s
                _time.sleep(2)
                status_resp = requests.get(f"{base_url}/status/{job_id}", headers=headers, timeout=10)
                status_data = status_resp.json()
                status = status_data.get("status")

                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    if isinstance(output, dict):
                        raw = output.get("generated_text", output.get("text", ""))
                    else:
                        raw = str(output)
                    return _unwrap_wm_response(raw)
                elif status == "FAILED":
                    logger.warning(f"WM_DEBUG: FAILED. sys_prompt_len={len(system_prompt)}")
                    return json.dumps({"error": "WM request failed"})
                # IN_QUEUE, IN_PROGRESS — keep polling

            logger.warning(f"WM_DEBUG: timeout after polling. sys_prompt_len={len(system_prompt)}")
            return json.dumps({"error": "WM timeout"})
        except Exception as e:
            logger.warning(f"WM_DEBUG: exception: {e}")
            return json.dumps({"error": str(e)})
        finally:
            _wm_semaphore.release()

    # Local WM endpoint — use session for connection pooling
    if not WM_ENDPOINT:
        return '{"error": "no WM endpoint"}'
    if not hasattr(call_world_model, '_session'):
        call_world_model._session = requests.Session()
    payload = {"input": {"state": state, "action": action if action else [],
                         "system_prompt": system_prompt, "max_tokens": 2048,
                         "num_diffusion_steps": 1}}
    try:
        resp = call_world_model._session.post(WM_ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("output", data).get("generated_text", "")
        return _unwrap_wm_response(raw)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from model output. Supports LFM2.5, Qwen, and raw JSON."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    text = re.sub(r"<\|/?tool\|>", "", text).strip()
    # 1. LFM2.5: <|tool_call_start|>[func(arg="val", arg2=123)]<|tool_call_end|>
    lfm_m = re.search(r'<\|tool_call_start\|>\s*\[(\w+)\((.*?)\)\]', text, re.DOTALL)
    if lfm_m:
        name = lfm_m.group(1)
        args = {}
        for kv in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|(\d+))', lfm_m.group(2)):
            args[kv.group(1)] = kv.group(2) if kv.group(2) is not None else int(kv.group(3))
        return {"name": name, "arguments": args}

    # 2. <tool_call>JSON</tool_call>, bare JSON object, or bare JSON array
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not m:
        # Try JSON array: [{"name": ...}]
        m = re.search(r'(\[\s*\{[^]]*"name"\s*:[^]]*\])', text, re.DOTALL)
    if not m:
        # Try single JSON object: {"name": ...}
        m = re.search(r'(\{[^}]*"name"\s*:[^}]*\})', text, re.DOTALL)
    if not m:
        return None
    try:
        import json_repair
        data = json_repair.loads(m.group(1))
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None
        name = str(data.get("name", ""))
        args = data.get("arguments", data.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {"name": name, "arguments": args}
    except:
        return None


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_api_name(tool_name: str) -> str:
    """Extract the API name from 'app__api_name' format."""
    if "__" in tool_name:
        return tool_name.split("__", 1)[1]
    return tool_name


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

class TokenCache:
    """Caches access tokens from login responses."""

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def cache_from_response(self, tool_name: str, response: str):
        app = tool_name.split("__")[0] if "__" in tool_name else None
        if not app:
            return
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                token = data.get("access_token") or data.get("token")
                if token:
                    self._cache[app] = str(token)[:40]
        except:
            pass

    def inject(self, tool_call: dict) -> dict:
        name = tool_call.get("name", "")
        app = name.split("__")[0] if "__" in name else None
        if not app or name.endswith("__login") or name.endswith("login"):
            return tool_call
        token = self._cache.get(app)
        if token:
            args = tool_call.get("arguments", {})
            if isinstance(args, dict):
                args["access_token"] = token
                tool_call["arguments"] = args
        return tool_call


# ---------------------------------------------------------------------------
# AppWorld Scheduler
# ---------------------------------------------------------------------------

class AppWorldScheduler(MultiTurnScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wm = bool(WM_ENDPOINT) or bool(RUNPOD_ENDPOINT_ID)
        self._token_caches = {}  # per-request token caches
        self._calls_by_request = {}  # per-request loop detection
        self._list_caches = {}  # per-request cache of full list responses for pagination
        self._logged_in_apps = {}  # per-request: req_id -> set of app names that have logged in

    def _get_task(self, req: RolloutInferRequest) -> dict:
        d = req.data_dict
        return {
            "instruction": d.get("instruction", ""),
            "ground_truth": d.get("ground_truth", ""),
            "wm_system_prompt": d.get("wm_system_prompt", ""),
            "max_turns": d.get("max_turns", 15),
        }

    def check_finished(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        stripped = strip_thinking(completion)

        if not stripped:
            return True

        tc = parse_tool_call(stripped)

        # Terminal: supervisor__complete_task
        if tc and "complete_task" in tc["name"]:
            return True

        # No valid tool call — give the model one retry before terminating
        if tc is None:
            req_id = getattr(infer_request, 'uuid', None) or id(infer_request)
            retry_key = f"_text_retries_{req_id}"
            if not hasattr(self, retry_key):
                setattr(self, retry_key, 0)
            retries = getattr(self, retry_key)
            if retries >= 1:
                return True  # Already retried once, terminate
            setattr(self, retry_key, retries + 1)
            # Don't terminate — step() will send a nudge message
            return False

        # Loop detection — use per-request state via uuid
        if tc:
            req_id = getattr(infer_request, 'uuid', None) or id(infer_request)
            if req_id not in self._calls_by_request:
                self._calls_by_request[req_id] = []
            recent = self._calls_by_request[req_id]
            sig = json.dumps({"name": tc["name"], "arguments": tc["arguments"]}, sort_keys=True)
            recent.append(sig)
            if len(recent) >= 2 and recent[-1] == recent[-2]:
                return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        stripped = strip_thinking(completion)
        task = self._get_task(infer_request)

        tc = parse_tool_call(stripped)
        if tc is None:
            infer_request.messages.append({
                "role": "user",
                "content": (
                    "You MUST respond with exactly one tool call and no prose. "
                    f"Output only this JSON array format: {CANONICAL_TOOL_FORMAT}"
                ),
            })
            return {"infer_request": infer_request}

        tool_name = tc["name"]
        tool_args = tc["arguments"]
        if tool_name not in VALID_TOOL_NAMES:
            infer_request.messages.append({
                "role": "user",
                "content": f"<tool_response>\n{{\"error\": \"invalid tool name: {tool_name}\"}}\n</tool_response>",
            })
            return {"infer_request": infer_request}

        # Auto-inject cached tokens (per-request cache)
        req_id = getattr(infer_request, 'uuid', None) or id(infer_request)
        if req_id not in self._token_caches:
            self._token_caches[req_id] = TokenCache()
        token_cache = self._token_caches[req_id]
        tc = token_cache.inject(tc)
        tool_args = tc["arguments"]

        # Strip access_token and cap page_limit before sending to WM
        wm_args = {k: v for k, v in tool_args.items() if k != "access_token"}
        if "page_limit" in wm_args:
            try:
                wm_args["page_limit"] = min(int(wm_args["page_limit"]), 20)
            except (ValueError, TypeError):
                wm_args["page_limit"] = 5

        # Build WM action
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        wm_action_msg = {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(wm_args)},
            }],
        }

        # Track per-request logged-in apps for auth enforcement
        if req_id not in self._logged_in_apps:
            self._logged_in_apps[req_id] = set()
        logged_in = self._logged_in_apps[req_id]

        # Intercept show_account_passwords — return creds from WM system prompt directly
        if tool_name == "supervisor__show_account_passwords":
            wm_prompt = task.get("wm_system_prompt", "")
            creds = []
            for m in re.finditer(r'- (\w+): username=([^,]+), password=(\S+)', wm_prompt):
                pwd = m.group(3)
                creds.append({"app": m.group(1), "username": m.group(2), "password": pwd})
            wm_response = json.dumps(creds) if creds else json.dumps({"error": "No credentials found in context"})
        # Call WM for everything else
        elif self.use_wm:
            wm_prompt = task.get("wm_system_prompt", "")
            guarded_response = None
            if os.environ.get("APPWORLD_WM_GUARD", "1") == "1":
                guarded_response = expected_appworld_response(wm_prompt, tool_name, wm_args, logged_in_apps=logged_in)
            if guarded_response is not None:
                wm_response = guarded_response
            else:
                wm_prompt = build_action_local_wm_prompt(wm_prompt, infer_request.messages, tool_name, wm_args)
                wm_response = call_world_model(wm_prompt, infer_request.messages, [wm_action_msg])
        else:
            wm_response = json.dumps({"status": "ok"})

        # Track successful logins
        if tool_name.endswith("__login") and "error" not in wm_response.lower():
            app = tool_name.split("__")[0]
            logged_in.add(app)

        # Enforce pagination on list responses
        page_index = tool_args.get("page_index", 0)
        page_limit = tool_args.get("page_limit", 5)
        try:
            page_index = int(page_index)
            page_limit = min(int(page_limit), 20)
        except (ValueError, TypeError):
            page_index, page_limit = 0, 5

        req_id = getattr(infer_request, 'uuid', None) or id(infer_request)
        cache_args = {
            k: v for k, v in wm_args.items()
            if k not in {"page_index", "page_limit", "access_token"}
        }
        cache_key = f"{req_id}:{tool_name}:{json.dumps(cache_args, sort_keys=True, default=str)}"

        if wm_response.strip().startswith("{"):
            try:
                import json_repair
                parsed = json_repair.loads(wm_response)
                if isinstance(parsed, dict):
                    # Find the array field
                    for key in parsed:
                        if isinstance(parsed[key], list) and len(parsed[key]) > 0:
                            # Cache the full list on first call for this API
                            if cache_key not in self._list_caches:
                                self._list_caches[cache_key] = {
                                    "key": key,
                                    "data": parsed[key],
                                    "total": parsed.get("total", len(parsed[key])),
                                }
                            cached = self._list_caches[cache_key]
                            total = cached["total"]
                            full_data = cached["data"]
                            start = page_index * page_limit
                            parsed[key] = full_data[start:start + page_limit]
                            parsed["total"] = total
                            wm_response = json.dumps(parsed)
                            break
            except Exception:
                pass

        # Cache tokens from login (per-request cache)
        if "login" in tool_name.lower():
            token_cache.cache_from_response(tool_name, wm_response)

        # Add tool response
        infer_request.messages.append({
            "role": "user",
            "content": f"<tool_response>\n{wm_response}\n</tool_response>",
        })

        # Log
        try:
            with open(TRAJECTORY_LOG, "a") as f:
                f.write(json.dumps({
                    "instruction": task["instruction"][:60],
                    "turn": current_turn,
                    "tool_call": {"name": tool_name, "args": {k: str(v)[:50] for k, v in tool_args.items()}},
                    "wm_response": wm_response[:500],
                }) + "\n")
        except Exception:
            pass

        return {"infer_request": infer_request}


multi_turns["appworld_scheduler"] = AppWorldScheduler


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class AppWorldReward(ORM):
    """
    WM-response-quality-based reward for AppWorld.

    Judges trajectories by:
    - WM response quality (success vs error)
    - Semantic alignment (did the model call the right tools for the task?)
    - Credential usage (real creds from show_account_passwords vs placeholders)
    - Placeholder/schema-as-args detection on ALL tools
    - Task completion (complete_task called after real work)
    - Stalling/giving-up penalties
    """

    # Semantic alignment: instruction keywords → expected/forbidden tool actions
    SEMANTIC_RULES = [
        # (instruction_keywords, forbidden_tool_keywords, expected_tool_keywords)
        (["reject", "deny", "decline"], ["accept"], []),
        (["delete", "remove"], ["add", "create", "like"], []),
        (["unlike", "unfollow"], ["like", "follow"], []),
        (["unfollow"], ["follow_artist"], []),
    ]

    # Placeholder patterns in tool call arg VALUES
    PLACEHOLDER_VALUE_PATTERNS = [
        "your_", "user's ", "your ", "my_", "user_",
        "{'type':", '{"type":', "new_playlist_id",
        "roommate1", "roommate2", "bill_amount",
        "your_username", "your_password", "your_venmo",
        "your_spotify", "user's spotify",
        "colleague_username", "colleague1", "correct_email@",
    ]

    def _format_multiplier(self, completion: str, parsed_tool_call: Optional[dict]) -> float:
        """Penalize parseable-but-noncanonical completions instead of rewarding them fully."""
        stripped = strip_thinking(completion or "").strip()
        if parsed_tool_call is None:
            return 0.0
        multiplier = 1.0
        # Detect native formats — don't penalize model-specific tool call syntax
        is_mistral = stripped.startswith("[TOOL_CALLS]")
        is_lfm = "<|tool_call_start|>" in stripped
        if "```" in stripped or "<tool_call>" in stripped or "<|tool|>" in stripped or "<|/tool|>" in stripped:
            multiplier *= 0.4
        if not is_mistral and not is_lfm:
            if '"arguments"' in stripped and '"parameters"' not in stripped:
                multiplier *= 0.7
            if not (stripped.startswith("[") or stripped.startswith("{")):
                multiplier *= 0.4
        return multiplier

    def _check_semantic_alignment(self, instruction: str, tool_names: list) -> float:
        """Check if the tools called align with the task instruction.
        Returns 1.0 if aligned, 0.3 if misaligned (heavy penalty)."""
        instr_lower = instruction.lower()
        for instr_keywords, forbidden, expected in self.SEMANTIC_RULES:
            if any(kw in instr_lower for kw in instr_keywords):
                for tool_name in tool_names:
                    tool_lower = tool_name.lower()
                    # Check forbidden: e.g., "reject" task calling "accept"
                    if any(f in tool_lower for f in forbidden):
                        return 0.3  # semantic mismatch penalty
        return 1.0

    def __call__(self, completions, **kwargs) -> List[float]:
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        messages_list = kwargs.get("messages", [[] for _ in completions])
        instructions = kwargs.get("instruction", [""] * len(completions))

        if os.environ.get("APPWORLD_REWARD_DEBUG") == "1":
            import logging
            _gt_sample = ground_truths[0] if ground_truths else "EMPTY"
            _inst_sample = instructions[0][:40] if instructions else "EMPTY"
            logging.warning(
                f"REWARD_DEBUG: gt='{_gt_sample}' inst='{_inst_sample}' "
                f"kwargs_keys={list(kwargs.keys())[:10]}"
            )

        rewards = []
        for i, completion in enumerate(completions):
            gt = ground_truths[i] if i < len(ground_truths) else ""
            gt = str(gt) if gt else ""
            messages = messages_list[i] if i < len(messages_list) else []
            instruction = instructions[i] if i < len(instructions) else ""

            # Build full text from all messages (assistant + tool responses) + completion
            full_text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    full_text += (msg.get("content") or "") + "\n"
            full_text += completion

            # Analyze each tool_response from the WM
            num_responses = 0
            num_success = 0
            num_error = 0
            num_empty = 0
            num_data = 0
            seen_calls = set()
            num_unique_calls = 0
            num_repeated_calls = 0
            used_real_creds = False
            used_placeholder_args = False  # FIX P1: track placeholders on ALL tools
            creds_available = False
            returned_creds = ""
            num_schema_args = 0
            all_tool_names = []  # track all tool names for semantic check
            complete_answers = []

            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                content = str(msg.get("content", ""))

                if msg.get("role") in ("user", "tool") and "<tool_response>" in content:
                    num_responses += 1
                    resp_match = re.search(r'<tool_response>\s*(.*?)\s*</tool_response>', content, re.DOTALL)
                    resp_text = resp_match.group(1) if resp_match else content

                    if not resp_text.strip():
                        num_empty += 1
                    elif '"error"' in resp_text.lower() or 'execution failed' in resp_text.lower() or 'traceback' in resp_text.lower():
                        num_error += 1
                    else:
                        num_success += 1
                        if len(resp_text) > 50 or '"id"' in resp_text or '"title"' in resp_text:
                            num_data += 1

                    # Track show_account_passwords
                    prev_assistant = ""
                    for k in range(j - 1, -1, -1):
                        prev = messages[k]
                        if isinstance(prev, dict) and prev.get("role") == "assistant":
                            prev_assistant = prev.get("content") or ""
                            break

                    if "show_account_passwords" in prev_assistant:
                        creds_available = True
                        returned_creds = resp_text

                    # Track call uniqueness
                    if prev_assistant:
                        call_sig = prev_assistant[:200]
                        if call_sig in seen_calls:
                            num_repeated_calls += 1
                        else:
                            num_unique_calls += 1
                        seen_calls.add(call_sig)

                # Check ALL tool calls for arg quality (FIX P1: not just login)
                if msg.get("role") == "assistant":
                    tc = parse_tool_call(strip_thinking(content))
                    if tc:
                        tool_name = tc.get("name", "")
                        all_tool_names.append(tool_name)
                        args = tc.get("arguments", {})
                        if tool_name == "supervisor__complete_task":
                            complete_answers.append(str(args.get("answer", "")))
                        arg_values_str = " ".join(str(v).lower() for v in args.values())

                        # Detect schema-as-args on ANY tool
                        if "{'type'" in arg_values_str or '{"type"' in arg_values_str:
                            num_schema_args += 1

                        # FIX P1: Check placeholder args on ALL tools (not just login)
                        if any(p in arg_values_str for p in self.PLACEHOLDER_VALUE_PATTERNS):
                            if "login" in tool_name:
                                used_placeholder_args = True  # placeholder login
                            elif args:  # non-login tool with placeholder args
                                used_placeholder_args = True

                        # Check login for real creds
                        if "login" in tool_name and creds_available and returned_creds:
                            for v in args.values():
                                v_str = str(v)
                                if len(v_str) > 3 and v_str in returned_creds:
                                    used_real_creds = True
                                    break

            # Also check completion for tool names
            final_tc = parse_tool_call(strip_thinking(completion))
            if final_tc:
                final_name = final_tc.get("name", "")
                all_tool_names.append(final_name)
                if final_name == "supervisor__complete_task":
                    complete_answers.append(str(final_tc.get("arguments", {}).get("answer", "")))

            final_format_multiplier = self._format_multiplier(completion, final_tc)
            called_complete = any(tool_name == "supervisor__complete_task" for tool_name in all_tool_names)
            has_any_tool_call = num_responses > 0 or final_tc is not None

            # Validate tool names against known valid set
            invalid_tool_count = sum(1 for t in all_tool_names if t not in VALID_TOOL_NAMES)
            invalid_tool_ratio = invalid_tool_count / max(len(all_tool_names), 1)

            # Extract answer for question tasks
            answer = complete_answers[-1] if complete_answers else ""

            # === QUESTION TASK ===
            if gt:
                if not called_complete:
                    q_reward = 0.0
                    if num_data > 0:
                        q_reward = 0.15
                    elif has_any_tool_call:
                        q_reward = 0.03
                elif gt.lower() in answer.lower():
                    q_reward = 1.0
                else:
                    q_reward = 0.0
                    # Numeric proximity for counting tasks
                    if gt.strip().isdigit():
                        # Extract numbers from the answer
                        answer_nums = re.findall(r'\d+', answer)
                        if answer_nums:
                            gt_val = int(gt.strip())
                            # Find closest number in answer
                            closest = min(answer_nums, key=lambda x: abs(int(x) - gt_val))
                            ratio = 1.0 - min(abs(int(closest) - gt_val) / max(gt_val, 1), 1.0)
                            # Keep partial numeric credit modest so plausible guesses do not dominate.
                            q_reward = max(q_reward, ratio * 0.25)
                        if num_data > 0:
                            q_reward = max(q_reward, 0.15)
                    else:
                        # Non-numeric question: flat partial credit for data
                        if num_data > 0:
                            q_reward = 0.15
                        elif num_success > 0:
                            q_reward = 0.08
                        elif has_any_tool_call:
                            q_reward = 0.03

                # Apply invalid tool penalty to question tasks too
                if invalid_tool_count > 0:
                    q_reward *= max(0.1, 1.0 - invalid_tool_ratio)
                # Penalize skipping credential workflow for questions too
                called_show_passwords_q = any("show_account_passwords" in t for t in all_tool_names)
                if not called_show_passwords_q and num_responses > 0:
                    q_reward *= 0.1
                if used_placeholder_args:
                    q_reward *= 0.2
                if num_schema_args > 0:
                    q_reward *= 0.3
                q_reward *= final_format_multiplier if called_complete else 1.0
                rewards.append(min(q_reward, 1.0))
                continue

            # === ACTION TASK ===
            if not has_any_tool_call:
                rewards.append(0.0)
                continue

            # Compute reward from WM response quality
            reward = 0.0

            if num_responses == 0:
                reward = 0.05
            else:
                success_rate = num_success / num_responses if num_responses > 0 else 0.0
                reward = success_rate * 0.6

                if num_data > 0:
                    reward += 0.1

            # Loop penalty
            total_calls = num_unique_calls + num_repeated_calls
            if total_calls > 2 and num_repeated_calls / total_calls > 0.5:
                reward *= 0.5

            # Credential quality — heavily penalize skipping the auth workflow
            called_show_passwords = any("show_account_passwords" in t for t in all_tool_names)
            if not called_show_passwords and num_responses > 0:
                reward *= 0.1  # near-zero if you skip getting credentials
            if used_real_creds:
                reward += 0.2
            if used_placeholder_args:  # FIX P1: applies to ALL tools now
                reward *= 0.2  # harsh penalty for fake credentials

            # Schema-as-args penalty
            if num_schema_args > 0:
                reward *= 0.3

            # Invalid tool name penalty
            if invalid_tool_count > 0:
                reward *= max(0.1, 1.0 - invalid_tool_ratio)

            # Penalty for giving up or stalling (FIX P2: handle both [ and { formats)
            completion_stripped = strip_thinking(completion)
            if not called_complete and num_responses > 0:
                if final_tc is None:
                    reward *= 0.3  # gave up with plain text
                elif final_tc is not None:
                    # Find where the tool call starts (support both [...] and {...})
                    tc_start = max(completion_stripped.rfind('['), completion_stripped.rfind('{'))
                    if tc_start > 0:
                        preamble = completion_stripped[:tc_start].strip()
                        if len(preamble) > 50:
                            reward *= 0.5  # stalling — text before tool call

            # Check if model called any action/mutation APIs (not just queries)
            QUERY_ONLY_TOOLS = {
                'supervisor__show_account_passwords', 'supervisor__complete_task',
                'spotify__login', 'spotify__show_playlist_library', 'spotify__show_playlist',
                'spotify__show_liked_songs', 'spotify__show_song_library', 'spotify__show_album',
                'spotify__show_album_library', 'spotify__show_recommendations', 'spotify__search_songs',
                'spotify__show_artist',
                'venmo__login', 'venmo__show_transactions', 'venmo__show_received_payment_requests',
                'venmo__show_social_feed',
                'phone__login', 'phone__search_text_messages', 'phone__search_voice_messages',
                'file_system__login', 'file_system__show_directory', 'file_system__show_file',
                'simple_note__login', 'simple_note__search_notes',
            }
            has_action_call = any(t not in QUERY_ONLY_TOOLS for t in all_tool_names)

            # complete_task: strong incentive to finish properly
            if called_complete:
                if has_action_call and num_success >= 1:
                    reward += 0.3  # did actual work + completed properly
                elif num_success >= 1:
                    reward += 0.1  # only queried, no actions — partial credit
                else:
                    reward += 0.05  # called complete but did nothing
            else:
                # Scale down reward without complete_task
                # Less harsh if model was actively working (has actions + many turns)
                if has_action_call and len(all_tool_names) >= 8:
                    reward *= 0.7  # actively working, ran out of turns
                else:
                    reward *= 0.5  # didn't try hard enough

            # FIX P0: Semantic alignment — APPLY LAST so it penalizes the final reward
            if instruction and all_tool_names:
                semantic_score = self._check_semantic_alignment(instruction, all_tool_names)
                if semantic_score < 1.0:
                    reward *= semantic_score

            reward *= final_format_multiplier if called_complete else 1.0
            rewards.append(min(reward, 1.0))

        return rewards


orms["appworld_reward"] = AppWorldReward
