"""WM prompt builder for ScienceWorld.

v7: Adds inventory injection, increased truncation, non-action output filtering.
Uses ## ENVIRONMENT STATE, ## TASK CONTEXT, ## DOMAIN RULES, ## STEERING DIRECTIVES,
and PREDICTION TARGET sections — the exact sections the WM was trained on.
"""
import re
from typing import Any, Optional, Set


def _extract_current_room(conversation_history: list) -> str:
    """Figure out which room the agent is currently in."""
    for msg in reversed(conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"You (?:move|teleport) to the (.+?)\.", content)
            if m:
                return m.group(1)
            m = re.search(r"(?:This room|This outside location) is called the (.+?)\.", content)
            if m:
                return m.group(1)
            if "You are in:" in content:
                m = re.search(r"is called the (.+?)\.", content)
                if m:
                    return m.group(1)
    return "unknown"


def _extract_room_contents(system_prompt: str, room_name: str) -> str:
    """Extract the room description block from the WM system prompt."""
    pattern = rf"\[{re.escape(room_name)}\]\n(.*?)(?=\n\n\[|\n\n##|\Z)"
    m = re.search(pattern, system_prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_task_description(system_prompt: str) -> str:
    m = re.search(r"Task: (.*?)(?:\n|$)", system_prompt)
    return m.group(1).strip() if m else "Unknown task"


def _extract_room_connections(system_prompt: str) -> str:
    m = re.search(r"## ROOM CONNECTIONS\n(.*?)(?=\n\n##|\Z)", system_prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def build_wm_prompt(system_prompt: str, conversation_history: list, action_text: str,
                    inventory: Optional[Set[str]] = None) -> str:
    """Build a WM prompt matching the training format of world_modeling_data_v5."""

    current_room = _extract_current_room(conversation_history)
    room_contents = _extract_room_contents(system_prompt, current_room) if current_room != "unknown" else ""
    task_desc = _extract_task_description(system_prompt)
    room_connections = _extract_room_connections(system_prompt)

    parts = []

    # == ENVIRONMENT STATE ==
    parts.append("## ENVIRONMENT STATE")
    parts.append("ScienceWorld is a text-based interactive science experiment environment with 10 rooms.")
    parts.append(f"The agent is currently in: {current_room}.")
    parts.append(f"Task: {task_desc}")
    parts.append("")

    if room_contents:
        parts.append(f"Visible objects in {current_room}:")
        parts.append(room_contents)
        parts.append("")

    if inventory:
        parts.append(f"Agent's inventory: {', '.join(sorted(inventory))}")
        parts.append("")

    if room_connections:
        parts.append("Room adjacency (for 'go to' validation):")
        parts.append(room_connections)
        parts.append("")

    # == TASK CONTEXT ==
    parts.append("## TASK CONTEXT")
    parts.append("The agent navigates rooms, picks up objects, and performs science experiments.")
    parts.append("Actions are plain text commands like 'teleport to kitchen', 'pick up glass cup', 'focus on water'.")
    parts.append(f"The active action is: {action_text}")
    parts.append("")

    # == DOMAIN RULES ==
    parts.append("## DOMAIN RULES")
    parts.append("- 'teleport to X' always succeeds. Response: 'You teleport to the X.'")
    parts.append("- 'go to X' only works if X is adjacent to the current room. If not adjacent: 'No known action matches that input.'")
    parts.append("- 'look around' returns a plain text list of objects in the current room. Format: 'This room is called X. In it, you see: obj1, obj2, obj3. You also see: door to Y, door to Z.'")
    parts.append("- 'pick up X' works only if X is visible in the current room AND has not already been picked up. Response: 'You move the X to the inventory.' If X is not present or already picked up: 'No known action matches that input.'")
    parts.append("- 'focus on X' works ONLY if X is explicitly listed in the current room description OR was previously picked up (in inventory). If X does NOT appear in the visible objects or inventory, respond: 'No known action matches that input.' NEVER confirm focus on an object that is not visibly present or in inventory — this is critical.")
    parts.append("- 'activate X' / 'deactivate X' works if X is in the current room. Response: 'The X is now activated/deactivated.'")
    parts.append("- 'put X in/on Y' works if X is in inventory. Response: 'You move the X to the Y.'")
    parts.append("- 'open X' / 'close X' works on doors and containers in the current room.")
    parts.append("- 'inventory' lists items the agent has picked up.")
    parts.append("- 'wait' / 'wait1' advances time. Response: '(1 tick passes)'")
    parts.append("- Invalid actions or objects not present: 'No known action matches that input.'")
    parts.append("- Items picked up earlier in the conversation are in inventory and no longer in their original room.")
    parts.append("")

    # == STEERING DIRECTIVES ==
    parts.append("## STEERING DIRECTIVES")
    parts.append("GROUNDING RULE: Your response MUST use ONLY objects, rooms, and substances that appear in the ENVIRONMENT STATE section above. Do NOT invent, hallucinate, or elaborate beyond what is listed. If the environment state lists 'a brick, a workbench', respond with EXACTLY those items — do not add tools, equipment, or descriptions that are not in the state.")
    parts.append("- For 'look around': copy the room description from the Visible objects section above VERBATIM. Do not add objects, do not elaborate, do not write prose. Just the object list.")
    parts.append("- For 'pick up X': check if X appears in the Visible objects. If yes: 'You move the X to the inventory.' If no: 'No known action matches that input.'")
    parts.append("- For 'focus on X': check Visible objects AND inventory. If X not found: 'No known action matches that input.'")
    parts.append("- Keep responses SHORT — one or two sentences maximum. Never write paragraphs, stories, or elaborate descriptions.")
    parts.append("- Output plain text only. No JSON, no markdown, no code, no arrays, no role tags.")
    parts.append("")

    parts.append(f"PREDICTION TARGET: The ScienceWorld observation text produced by the action '{action_text}' given the current environment state.")

    return "\n".join(parts)


# Patterns that indicate non-action narrative output
_NARRATIVE_RE = re.compile(
    r'^(?:success|fail|error|done|complete[d]?|score|'
    r'i need|i should|i will|i\'ll|let me|the task|this task|now i|'
    r'step \d|observation|reward|result|note:)',
    re.IGNORECASE,
)

# Valid ScienceWorld action prefixes
_ACTION_PREFIXES = (
    'look around', 'look at', 'go to', 'teleport to', 'pick up',
    'put ', 'open ', 'close ', 'activate ', 'deactivate ',
    'focus on', 'pour ', 'use ', 'mix ', 'connect ', 'wait',
    'inventory', 'read ', 'eat ', 'drink ', 'drop ', 'examine ',
    'move ', 'push ', 'pull ', 'turn ', 'task',
)


def parse_action(text: str) -> str:
    """Extract a clean action from model output. Handles Qwen, Mistral, and LFM formats."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = re.sub(r"\[/?INST\]|\[TOOL_CALLS\]|\[AVAILABLE_TOOLS\]|\[/AVAILABLE_TOOLS\]|\[TOOL_RESULTS\]|\[/TOOL_RESULTS\]", "", text).strip()
    text = text.strip('"\'')
    text = text.split("\n")[0].strip()
    text = re.sub(r"^(?:Action|Command|>)\s*:?\s*", "", text, flags=re.IGNORECASE).strip()

    if not text:
        return ""
    # Reject pure numbers (agent outputting scores)
    if re.match(r'^-?\d+\.?\d*$', text):
        return ""
    # Reject very long outputs (not valid ScienceWorld actions)
    if len(text) > 200:
        return ""
    # Reject obvious narrative/status text that isn't a valid action
    if _NARRATIVE_RE.match(text) and not any(text.lower().startswith(p) for p in _ACTION_PREFIXES):
        return ""
    return text
