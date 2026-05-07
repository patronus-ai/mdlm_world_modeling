"""ALFWorld WM prompt builder + deterministic local response handler.

Mirrors sciworld_wm_prompt.py and appworld_wm_prompt.py. Two responsibilities:

1. `expected_alfworld_response(...)` — deterministic stand-in for the trained WM.
   Handles every ALFWorld action verb whose text feedback can be derived from a
   small running state (inventory, current location, receptacle contents/state,
   object_states). The text matches the strings the textworld engine emits
   verbatim, e.g. 'You move the pot 1 to the shelf 1.', 'You arrive at cabinet 8. The cabinet 8 is closed.'.

2. `build_alfworld_wm_prompt(...)` — for the residual cases (mainly: contents of
   a previously-unopened receptacle, free-form examine), build a SciWorld-style
   action-local prompt (## ENVIRONMENT STATE / ## TASK CONTEXT / ## DOMAIN RULES
   / ## STEERING DIRECTIVES / PREDICTION TARGET) for the SDAR diffusion WM.

The wm_system_prompt baked per task contains:
  - 'Task: <goal text>'
  - 'Task type: <pick_and_place_simple | ...>'
  - 'Goal predicates: target=..., destination=..., clean/heat/cool/slice=..., toggle=...'
  - '## INITIAL SCENE' — list of receptacles (with open/closed state) + initial visible objects
  - '## INITIAL OBJECT LOCATIONS' (optional) — known (object -> receptacle) facts

Goal completion is evaluated by `check_goal_satisfied()` from the conversation
history alone, so reward never needs the real env at training time.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# -- Action verbs we know about ----------------------------------------------------

# Each action grammar is conservative: ALFWorld is case-sensitive lowercase.
# Each pattern accepts both well-formed actions and the slightly-degraded forms
# the trained agent often produces: missing 'from'/'to' clause, missing instance
# number, missing trailing punctuation. Group names mark whether the field is
# present; missing fields default to None and get inferred from state at call time.
ACTION_PATTERNS = {
    "look":      re.compile(r"^look$", re.I),
    "inventory": re.compile(r"^inventory$", re.I),
    "help":      re.compile(r"^help$", re.I),
    "go":        re.compile(r"^go to (?P<target>[\w ]+?\d*)$", re.I),
    "open":      re.compile(r"^open (?P<target>[\w ]+?\d*)$", re.I),
    "close":     re.compile(r"^close (?P<target>[\w ]+?\d*)$", re.I),
    # take: 'take obj from src' OR 'take obj' (src inferred from current_loc)
    "take":      re.compile(r"^take (?P<obj>[\w ]+?\d*)(?: from (?P<src>[\w ]+?\d*))?$", re.I),
    # put/move: 'put X in/on Y', 'move X to Y' OR 'put/move X' (dst from current_loc)
    "put":       re.compile(r"^(?:put|move) (?P<obj>[\w ]+?\d*?)(?: (?:in|on|to|in/on) (?P<dst>[\w ]+?\d*))?$", re.I),
    "examine":   re.compile(r"^(?:examine|look at) (?P<target>[\w ]+?\d*)$", re.I),
    "clean":     re.compile(r"^clean (?P<obj>[\w ]+?\d*)(?: with (?P<tool>[\w ]+?\d*))?$", re.I),
    "heat":      re.compile(r"^heat (?P<obj>[\w ]+?\d*)(?: with (?P<tool>[\w ]+?\d*))?$", re.I),
    "cool":      re.compile(r"^cool (?P<obj>[\w ]+?\d*)(?: with (?P<tool>[\w ]+?\d*))?$", re.I),
    "slice":     re.compile(r"^slice (?P<obj>[\w ]+?\d*)(?: with (?P<tool>[\w ]+?\d*))?$", re.I),
    "use":       re.compile(r"^use (?P<target>[\w ]+?\d*)$", re.I),
}


def _resolve_instance(name: str | None, candidates: list[str]) -> str | None:
    """Resolve a class-only name like 'apple' to 'apple N' from candidates.
    When ambiguous (multiple matches), picks the first rather than failing."""
    if not name:
        return None
    name = name.strip()
    name = re.sub(r"^(?:the |a |an )", "", name, flags=re.I).strip()
    # Exact match with instance number
    if re.match(r"^[\w ]+\d+$", name):
        if name in candidates:
            return name
        # Fuzzy: same class, different number (e.g. "pot 1" matches "pot 2" if pot 1 not in list)
        base = name.rstrip(" 0123456789").strip()
        for c in candidates:
            if c.rstrip(" 0123456789").strip() == base:
                return c
        return None
    # Class-only: find matches
    matches = [c for c in candidates if c.startswith(name + " ") or c == name]
    if matches:
        return matches[0]  # pick first — better than returning None
    return None

NOTHING_HAPPENS = "Nothing happens."

# Verbatim textworld help text — the env emits this for the `help` command.
HELP_TEXT = (
    "Available commands:\n"
    "  look:                             look around your current location\n"
    "  inventory:                        check your current inventory\n"
    "  go to (receptacle):               move to a receptacle\n"
    "  open (receptacle):                open a receptacle\n"
    "  close (receptacle):               close a receptacle\n"
    "  take (object) from (receptacle):  take an object from a receptacle\n"
    "  move (object) to (receptacle):  place an object in or on a receptacle\n"
    "  examine (something):              examine a receptacle or an object\n"
    "  use (object):                     use an object\n"
    "  heat (object) with (receptacle):  heat an object using a receptacle\n"
    "  clean (object) with (receptacle): clean an object using a receptacle\n"
    "  cool (object) with (receptacle):  cool an object using a receptacle\n"
    "  slice (object) with (object):     slice an object using a sharp object\n"
)


# -- Goal extraction ---------------------------------------------------------------

# Goal predicates we track per task_type. Field meanings:
#  target_obj:     object class to manipulate (lowercase, e.g. 'handtowel')
#  destination:    parent receptacle class (e.g. 'garbagecan')
#  mrecep:         intermediate movable receptacle (pick_two — usually empty)
#  toggle_target:  receptacle whose 'use' must fire (e.g. 'desklamp')
#  needs_clean:    must clean target before placing
#  needs_heat:     must heat target before placing
#  needs_cool:     must cool target before placing
#  needs_two:      two distinct instances of target_obj must end up at destination
def parse_task_metadata(task_type: str, pddl_params: Dict[str, str]) -> Dict[str, Any]:
    target_obj  = (pddl_params.get("object_target")  or "").lower()
    destination = (pddl_params.get("parent_target")  or "").lower()
    mrecep      = (pddl_params.get("mrecep_target")  or "").lower()
    toggle      = (pddl_params.get("toggle_target")  or "").lower()
    return {
        "task_type": task_type,
        "target_obj": target_obj,
        "destination": destination,
        "mrecep": mrecep,
        "toggle_target": toggle,
        "needs_clean": task_type == "pick_clean_then_place_in_recep",
        "needs_heat":  task_type == "pick_heat_then_place_in_recep",
        "needs_cool":  task_type == "pick_cool_then_place_in_recep",
        "needs_two":   task_type == "pick_two_obj_and_place",
        "needs_toggle": task_type == "look_at_obj_in_light",
    }


# -- Lightweight running state -----------------------------------------------------

class AlfState:
    """Tracks the minimum facts needed to render deterministic feedback.

    Populated from the wm_system_prompt at construction, then advanced by
    `apply(action)` after every parsed action."""

    def __init__(self, wm_system_prompt: str):
        self.current_loc: Optional[str] = None     # 'cabinet 8' etc; None = mid-room
        self.inventory: List[str] = []             # object instances ('pot 1')
        self.opened: set[str] = set()              # receptacles the agent has opened
        self.closed: set[str] = set()              # receptacles known closed (initial state)
        self.contents: Dict[str, List[str]] = {}   # receptacle -> [objects there now]
        self.obj_state: Dict[str, set] = {}        # object -> {'clean','hot','cool','sliced','on'}
        self.receptacles: List[str] = []           # all known receptacles
        self.objects_seen: set[str] = set()        # objects we've heard mentioned
        self._parse_initial(wm_system_prompt)

    def _parse_initial(self, sys_prompt: str):
        # Receptacles + state from "## INITIAL SCENE". We accept open/closed/unknown.
        scene = _section(sys_prompt, "## INITIAL SCENE")
        for line in scene.splitlines():
            m = re.match(r"-\s+([\w ]+\d+)\s*\((open|closed|unknown)\)", line.strip())
            if m:
                recep, st = m.group(1).strip(), m.group(2)
                self.receptacles.append(recep)
                if st == "closed":
                    self.closed.add(recep)
                elif st == "open":
                    self.opened.add(recep)
        # Object locations from "## INITIAL OBJECT LOCATIONS"
        locs = _section(sys_prompt, "## INITIAL OBJECT LOCATIONS")
        for line in locs.splitlines():
            m = re.match(r"-\s+([\w ]+\d+)\s*:\s*(?:in|on)\s+([\w ]+\d+)", line.strip())
            if m:
                obj, recep = m.group(1).strip(), m.group(2).strip()
                self.contents.setdefault(recep, []).append(obj)
                self.objects_seen.add(obj)

    # ---- helpers ----
    def is_closed(self, recep: str) -> bool:
        return recep in self.closed and recep not in self.opened

    def see_at(self, recep: str) -> List[str]:
        return list(self.contents.get(recep, []))

    def has(self, obj: str) -> bool:
        return obj in self.inventory

    def state_of(self, obj: str) -> set:
        return self.obj_state.setdefault(obj, set())


# -- Section extraction helper -----------------------------------------------------

def _section(text: str, header: str) -> str:
    pat = rf"^{re.escape(header)}\s*$(.*?)(?=^##\s|\Z)"
    m = re.search(pat, text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


# -- Conversation-history reconstruction ------------------------------------------

# Patterns we read back from <tool_response> (or plain user) messages to keep state
# consistent across turns when the plugin restores from messages instead of a live state object.
_RESP_GO_OPEN     = re.compile(r"You arrive at ([\w ]+\d+)\. On the \1, you see (.+?)\.", re.S)
_RESP_GO_NOTHING  = re.compile(r"You arrive at ([\w ]+\d+)\. On the \1, you see nothing\.")
_RESP_GO_CLOSED   = re.compile(r"You arrive at ([\w ]+\d+)\. The \1 is closed\.")
_RESP_OPEN        = re.compile(r"You open the ([\w ]+\d+)\. The \1 is open\. In it, you see (.+?)\.", re.S)
_RESP_OPEN_EMPTY  = re.compile(r"You open the ([\w ]+\d+)\. The \1 is open\. In it, you see nothing\.")
_RESP_CLOSE       = re.compile(r"You close the ([\w ]+\d+)\.")
_RESP_TAKE        = re.compile(r"You pick up the ([\w ]+\d+) from the ([\w ]+\d+)\.")
_RESP_PUT         = re.compile(r"You (?:move|put) the ([\w ]+\d+) (?:to|in/on|in|on) the ([\w ]+\d+)\.")
_RESP_CLEAN       = re.compile(r"You clean the ([\w ]+\d+) using the ([\w ]+\d+)\.")
_RESP_HEAT        = re.compile(r"You heat the ([\w ]+\d+) using the ([\w ]+\d+)\.")
_RESP_COOL        = re.compile(r"You cool the ([\w ]+\d+) using the ([\w ]+\d+)\.")
_RESP_SLICE       = re.compile(r"You slice the ([\w ]+\d+) using the ([\w ]+\d+)\.")
_RESP_USE         = re.compile(r"You turn on the ([\w ]+\d+)\.")


def _split_listed(items: str) -> List[str]:
    items = items.strip()
    items = re.sub(r"^a |^an ", "", items)
    chunks = re.split(r",\s*(?:and )?| and ", items)
    out = []
    for c in chunks:
        c = c.strip()
        c = re.sub(r"^a |^an ", "", c)
        if c:
            out.append(c)
    return out


def update_state_from_message(state: AlfState, content: str) -> None:
    """Replay a single tool/user message's observation text into `state`."""
    # Parse "go to" responses — including variant where textworld drops the number
    # in "On the stoveburner, you see" vs "On the stoveburner 1, you see"
    for m in _RESP_GO_CLOSED.finditer(content):
        recep = m.group(1)
        state.current_loc = recep
        state.closed.add(recep)
        state.opened.discard(recep)
        if recep not in state.receptacles:
            state.receptacles.append(recep)
    for m in _RESP_GO_OPEN.finditer(content):
        recep, items = m.group(1), m.group(2)
        state.current_loc = recep
        state.opened.add(recep)
        state.closed.discard(recep)
        if recep not in state.receptacles:
            state.receptacles.append(recep)
        objs = _split_listed(items)
        state.contents[recep] = list(dict.fromkeys(state.contents.get(recep, []) + objs))
        for o in objs:
            state.objects_seen.add(o)
    # Catch WM responses where the receptacle name in "On the X" doesn't exactly match
    # "You arrive at Y" — e.g. "You arrive at stoveburner 1. On the stoveburner, you see..."
    # or "You arrive at desk 1. On the table, you see..."
    for m in re.finditer(r"You arrive at ([\w ]+\d+)\. On the [^,]+, you see (.+?)\.", content, re.S):
        recep = m.group(1)
        items_text = m.group(2).strip()
        # Skip if already matched by the strict regex above
        if _RESP_GO_OPEN.search(content) and recep == (_RESP_GO_OPEN.search(content).group(1)):
            continue
        state.current_loc = recep
        state.opened.add(recep)
        state.closed.discard(recep)
        if recep not in state.receptacles:
            state.receptacles.append(recep)
        if items_text.lower() != "nothing":
            objs = _split_listed(items_text)
            # Add instance numbers to objects that lack them, using objects_seen as reference
            resolved_objs = []
            for o in objs:
                if re.match(r"^[\w ]+\d+$", o):
                    resolved_objs.append(o)
                else:
                    # Try to find a matching instance in objects_seen
                    matches = [s for s in state.objects_seen if s.startswith(o + " ")]
                    if len(matches) == 1:
                        resolved_objs.append(matches[0])
                    else:
                        resolved_objs.append(o)  # keep as-is
            state.contents[recep] = list(dict.fromkeys(state.contents.get(recep, []) + resolved_objs))
            for o in resolved_objs:
                state.objects_seen.add(o)
        else:
            state.contents[recep] = []
    # Also match "nothing" variant without the strict pattern
    for m in re.finditer(r"You arrive at ([\w ]+\d+)\. On the [^,]+, you see nothing\b", content):
        recep = m.group(1)
        state.current_loc = recep
        if recep not in state.receptacles:
            state.receptacles.append(recep)
        if recep not in state.contents:
            state.contents[recep] = []
    for m in _RESP_GO_NOTHING.finditer(content):
        recep = m.group(1)
        state.current_loc = recep
        state.contents[recep] = []
        if recep not in state.receptacles:
            state.receptacles.append(recep)
    for m in _RESP_OPEN.finditer(content):
        recep, items = m.group(1), m.group(2)
        state.opened.add(recep); state.closed.discard(recep)
        objs = _split_listed(items)
        state.contents[recep] = list(dict.fromkeys(state.contents.get(recep, []) + objs))
        for o in objs:
            state.objects_seen.add(o)
    for m in _RESP_OPEN_EMPTY.finditer(content):
        recep = m.group(1)
        state.opened.add(recep); state.closed.discard(recep)
        state.contents[recep] = []
    for m in _RESP_CLOSE.finditer(content):
        recep = m.group(1)
        # Re-classify as closed; preserve known contents for next open.
        if recep in state.opened:
            state.opened.discard(recep)
            state.closed.add(recep)
    for m in _RESP_TAKE.finditer(content):
        obj, src = m.group(1), m.group(2)
        if obj not in state.inventory:
            state.inventory.append(obj)
        if obj in state.contents.get(src, []):
            state.contents[src].remove(obj)
        state.objects_seen.add(obj)
    for m in _RESP_PUT.finditer(content):
        obj, dst = m.group(1), m.group(2)
        if obj in state.inventory:
            state.inventory.remove(obj)
        state.contents.setdefault(dst, [])
        if obj not in state.contents[dst]:
            state.contents[dst].append(obj)
        state.objects_seen.add(obj)
    for m in _RESP_CLEAN.finditer(content):
        state.state_of(m.group(1)).add("clean")
    for m in _RESP_HEAT.finditer(content):
        state.state_of(m.group(1)).add("hot")
    for m in _RESP_COOL.finditer(content):
        state.state_of(m.group(1)).add("cool")
    for m in _RESP_SLICE.finditer(content):
        state.state_of(m.group(1)).add("sliced")
    for m in _RESP_USE.finditer(content):
        state.state_of(m.group(1)).add("on")


def replay_state(wm_system_prompt: str, messages: List[Dict[str, Any]]) -> AlfState:
    state = AlfState(wm_system_prompt)
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("user", "tool"):
            update_state_from_message(state, str(msg.get("content") or ""))
    return state


# -- Admissible commands generator -------------------------------------------------

def admissible_commands(state: "AlfState") -> List[str]:
    """Generate the admissible commands for the current state, mirroring textworld's."""
    cmds: list[str] = []
    cur = state.current_loc
    inv = list(state.inventory)
    here_items = list(state.see_at(cur)) if cur else []

    # always-valid utility verbs
    cmds.append("look")
    cmds.append("inventory")
    cmds.append("help")

    # navigation
    for r in state.receptacles:
        cmds.append(f"go to {r}")

    # at a receptacle: examine it
    if cur:
        cmds.append(f"examine {cur}")

    # examine items in inventory and at current loc
    for o in inv + here_items:
        cmds.append(f"examine {o}")

    # open/close — only for receptacles whose CLASS is closable.
    OPENABLE = {"cabinet", "drawer", "fridge", "microwave", "safe", "box"}
    cls = cur.split()[0].lower() if cur else ""
    if cur and cls in OPENABLE:
        if state.is_closed(cur):
            cmds.append(f"open {cur}")
        else:
            cmds.append(f"close {cur}")

    # take from current loc
    if cur and not state.is_closed(cur):
        for o in here_items:
            cmds.append(f"take {o} from {cur}")

    # put/move from inventory to current loc
    if cur and not state.is_closed(cur):
        for o in inv:
            cmds.append(f"move {o} to {cur}")

    # transformations only when holding something AND at a matching appliance
    if cur and inv:
        if "sinkbasin" in cur:
            for o in inv:
                cmds.append(f"clean {o} with {cur}")
        if "microwave" in cur:
            for o in inv:
                cmds.append(f"heat {o} with {cur}")
        if "fridge" in cur:
            for o in inv:
                cmds.append(f"cool {o} with {cur}")

    # use lamps / objects at current loc
    for o in here_items + inv:
        if "lamp" in o or "lightswitch" in o:
            cmds.append(f"use {o}")

    # slice with knife
    if cur and any("knife" in o for o in inv):
        knife = next(o for o in inv if "knife" in o)
        for o in here_items + [i for i in inv if "knife" not in i]:
            cmds.append(f"slice {o} with {knife}")

    # de-dup, preserve order
    seen = set(); out = []
    for c in cmds:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


# -- Deterministic response handler -----------------------------------------------

def _parse_action(action_text: str) -> Optional[Tuple[str, Dict[str, str]]]:
    a = (action_text or "").strip().rstrip(".")
    # Normalize: strip garbage prefixes (may repeat), Action: prefix, lowercase
    a = re.sub(r"(?:<<[^>]*>>\s*)+", "", a).strip()
    a = re.sub(r"^Action:\s*", "", a, flags=re.I).strip()
    a = a.lower()
    for verb, pat in ACTION_PATTERNS.items():
        m = pat.match(a)
        if m:
            args = {}
            for k, v in m.groupdict().items():
                if v is None:
                    args[k] = None
                else:
                    # Strip leading article ("the ", "a ", "an ") from any name slot.
                    args[k] = re.sub(r"^(?:the |a |an )", "", v.strip(), flags=re.I).strip()
            return verb, args
    return None


def _format_listed(objects: List[str]) -> str:
    if not objects:
        return "nothing"
    # textworld lists items alphabetically by class then in DESCENDING index order
    # (e.g. "potato 3, potato 2", "statue 3, statue 2, statue 1").
    sorted_objs = sorted(
        objects,
        key=lambda o: (o.split()[0], -int(o.split()[-1]) if o.split()[-1].isdigit() else 0),
    )
    items = [f"a {o}" for o in sorted_objs]
    if len(items) == 1:
        return items[0]
    # textworld uses Oxford comma for 2+ items: "a X, and a Y" / "a X, a Y, and a Z".
    return ", ".join(items[:-1]) + ", and " + items[-1]


def expected_alfworld_response(
    wm_system_prompt: str,
    messages: List[Dict[str, Any]],
    action_text: str,
) -> Optional[str]:
    """Return the textworld-style observation string when we can derive it locally.

    Returns None when the answer depends on knowledge we don't have (mainly:
    contents of a closed receptacle that wasn't pre-populated in the system
    prompt, or open-ended `examine` text). Caller falls back to the WM in that
    case.
    """
    parsed = _parse_action(action_text)
    if parsed is None:
        return NOTHING_HAPPENS

    verb, args = parsed
    state = replay_state(wm_system_prompt, messages)

    # ---- look ----
    # textworld emits the full receptacle list only in the reset welcome message;
    # a plain mid-room `look` returns 'you see nothing'. After moving to a
    # receptacle, the engine often includes adjacent receptacles in the description
    # ('you are facing shelf 2, and shelf 1'), which we can't reconstruct — defer
    # to the WM in that case.
    if verb == "look":
        if state.current_loc is None:
            return "You are in the middle of a room. Looking quickly around you, you see nothing."
        return None  # let WM render "facing X, and Y. Next to it, you see ..."

    # ---- help ----
    if verb == "help":
        return HELP_TEXT.rstrip()

    # ---- inventory ----
    if verb == "inventory":
        if not state.inventory:
            return "You are not carrying anything."
        return f"You are carrying: {_format_listed(state.inventory)}."

    # ---- go to ----
    if verb == "go":
        recep = args["target"]
        # Accept any receptacle-shaped name (word + digit) even if not in initial list
        is_recep_shaped = bool(re.match(r"^[\w ]+\d+$", recep))
        if not is_recep_shaped:
            return NOTHING_HAPPENS
        # Track as known receptacle if new
        if recep not in state.receptacles:
            state.receptacles.append(recep)
        state.current_loc = recep
        if state.is_closed(recep):
            return f"You arrive at {recep}. The {recep} is closed."
        # If we've observed contents, render verbatim.
        if recep in state.contents:
            items = state.see_at(recep)
            if not items:
                return f"You arrive at {recep}. On the {recep}, you see nothing."
            return f"You arrive at {recep}. On the {recep}, you see {_format_listed(items)}."
        # Receptacle in our known list but no contents recorded — it was empty during
        # the initial scene walk. Return "see nothing" instead of deferring to WM.
        if recep in state.receptacles:
            return f"You arrive at {recep}. On the {recep}, you see nothing."
        # Truly unknown receptacle (not in initial list, not visited before).
        # Defer to WM for contents.
        return None

    # ---- open ----
    if verb == "open":
        recep = args["target"]
        # Only cabinets, drawers, fridges, microwaves, safes, boxes are openable
        openable_classes = ("cabinet", "drawer", "fridge", "microwave", "safe", "box")
        recep_class = recep.rstrip(" 0123456789").strip()
        if recep_class not in openable_classes:
            return NOTHING_HAPPENS  # certain: not openable
        if recep in state.receptacles and not state.is_closed(recep):
            return NOTHING_HAPPENS  # certain: already open
        items = state.see_at(recep)
        if items:
            state.opened.add(recep)
            state.closed.discard(recep)
            return f"You open the {recep}. The {recep} is open. In it, you see {_format_listed(items)}."
        # closed receptacle with unknown contents — defer to WM
        return None

    # ---- close ----
    if verb == "close":
        recep = args["target"]
        if recep not in state.opened:
            return NOTHING_HAPPENS  # not open -> can't close
        state.opened.discard(recep)
        state.closed.add(recep)
        return f"You close the {recep}."

    # ---- take ----
    if verb == "take":
        src = args.get("src") or state.current_loc
        if not src:
            return None  # defer — don't know where we are
        if state.current_loc and state.current_loc != src:
            return NOTHING_HAPPENS  # certain: wrong location
        # Resolve obj against known contents at src
        obj = _resolve_instance(args.get("obj"), state.see_at(src))
        if obj is None:
            obj = _resolve_instance(args.get("obj"), list(state.objects_seen))
        if obj is None:
            # Don't know if object is there — defer to WM
            return None
        return f"You pick up the {obj} from the {src}."

    # ---- put / move ----
    if verb == "put":
        dst = args.get("dst") or state.current_loc
        if not dst:
            return None  # defer
        if state.current_loc and state.current_loc != dst:
            return NOTHING_HAPPENS  # certain: wrong location
        if state.is_closed(dst):
            return NOTHING_HAPPENS  # certain: closed
        obj = _resolve_instance(args.get("obj"), state.inventory)
        if obj is None:
            return None  # defer — maybe inventory tracking is off
        return f"You move the {obj} to the {dst}."

    # ---- clean / heat / cool ----
    for v, tool_kind, verb_text in [
        ("clean", "sinkbasin", "clean"),
        ("heat",  "microwave", "heat"),
        ("cool",  "fridge",    "cool"),
    ]:
        if verb == v:
            tool = args.get("tool")
            if not tool and state.current_loc and tool_kind in state.current_loc:
                tool = state.current_loc
            if not tool or tool_kind not in tool:
                return None  # defer — don't know the tool
            # Don't require exact location match — agent may be at the tool
            # even if state.current_loc doesn't match (tracking gap)
            if state.current_loc and tool_kind not in state.current_loc:
                return None  # defer — might be a tracking issue
            obj = _resolve_instance(args.get("obj"), state.inventory)
            if obj is None:
                return None  # defer — inventory tracking may be off
            return f"You {verb_text} the {obj} using the {tool}."

    # ---- slice ----
    if verb == "slice":
        candidates = state.see_at(state.current_loc) + state.inventory if state.current_loc else state.inventory
        obj = _resolve_instance(args.get("obj"), candidates)
        tool = args.get("tool")
        if not tool:
            tool = next((i for i in state.inventory if "knife" in i), None)
        if obj is None or not tool or "knife" not in tool or tool not in state.inventory:
            return NOTHING_HAPPENS
        return f"You slice the {obj} using the {tool}."

    # ---- use ----
    if verb == "use":
        target_name = args.get("target", "")
        # Desklamps/floorlamps — if the name contains "lamp", just confirm
        if "lamp" in target_name:
            resolved = _resolve_instance(target_name, list(state.objects_seen))
            if resolved:
                return f"You turn on the {resolved}."
            # Even if not seen, if it looks like "desklamp N", confirm it
            if re.match(r"^[\w]+lamp \d+$", target_name):
                return f"You turn on the {target_name}."
        if state.current_loc is None:
            return None
        here = state.see_at(state.current_loc)
        target = _resolve_instance(target_name, here + state.inventory + list(state.objects_seen))
        if target is None:
            return NOTHING_HAPPENS  # not a usable object
        return f"You turn on the {target}."

    # ---- examine ----
    # textworld allows examining any visible receptacle/object — phrasing
    # depends on the entity. Always defer to the WM.
    if verb == "examine":
        return None

    return None


# -- WM prompt builder (residual cases) ------------------------------------------

def build_alfworld_wm_prompt(
    wm_system_prompt: str,
    messages: List[Dict[str, Any]],
    action_text: str,
) -> str:
    """Build a SciWorld-style action-local prompt for the SDAR WM.

    Mirrors sciworld_wm_prompt.build_wm_prompt sections:
      ## ENVIRONMENT STATE -> identity, task, goal, current loc, inventory, full
                              receptacle map (open/closed/contents), object_state
      ## TASK CONTEXT      -> what to predict, the active action
      ## DOMAIN RULES      -> exact verb semantics + verbatim response phrasings
      ## STEERING DIRECTIVES -> safety constraints / formatting
      PREDICTION TARGET     -> single-line directive
    """

    state = replay_state(wm_system_prompt, messages)
    task_line = re.search(r"^Task:\s*(.+)$", wm_system_prompt, re.MULTILINE)
    task = task_line.group(1).strip() if task_line else "(unknown)"
    task_type_line = re.search(r"^Task type:\s*(.+)$", wm_system_prompt, re.MULTILINE)
    task_type = task_type_line.group(1).strip() if task_type_line else "(unknown)"
    goal_line = re.search(r"^Goal predicates:\s*(.+)$", wm_system_prompt, re.MULTILINE)
    goal_pred = goal_line.group(1).strip() if goal_line else ""

    parts: List[str] = []

    # =============================================================
    # ## ENVIRONMENT STATE
    # =============================================================
    parts.append("## ENVIRONMENT STATE")
    parts.append(
        "ALFWorld is a TextWorld household environment derived from ALFRED. The agent operates "
        "in a single room (kitchen / bathroom / bedroom / living-room / office) populated with "
        "receptacles (cabinet, drawer, fridge, microwave, countertop, shelf, sidetable, "
        "diningtable, desk, dresser, bed, sinkbasin, stoveburner, toilet, bathtub, garbagecan, "
        "etc.) each carrying a 1-indexed instance id (e.g. 'cabinet 8'). Objects are also "
        "1-indexed within their class (e.g. 'apple 1', 'mug 2'). All names are lowercase."
    )
    parts.append(f"Task: {task}")
    parts.append(f"Task type: {task_type}")
    if goal_pred:
        parts.append(f"Goal predicates: {goal_pred}")
    parts.append(f"Current location: {state.current_loc or 'middle of the room (no receptacle in front)'}")
    if state.inventory:
        parts.append(f"Inventory: {', '.join(state.inventory)}")
    else:
        parts.append("Inventory: (empty)")
    parts.append("")

    # Receptacle table — every known receptacle, current open/closed status.
    parts.append("Receptacles in this room (all 1-indexed, lowercase):")
    for r in state.receptacles:
        if state.is_closed(r):
            tag = "currently CLOSED — must be opened to take/place inside"
        elif r in state.opened:
            tag = "OPEN (was closed; agent has opened it)"
        else:
            tag = "open / not openable (countertop, table, shelf, sinkbasin, stoveburner, etc.)"
        parts.append(f"- {r}: {tag}")
    parts.append("")

    # Known contents — both pre-baked from initial scene walk AND observed during rollout.
    if state.contents:
        parts.append("Known receptacle contents (pre-baked from initial scene walk + updates from "
                     "this rollout's take/put events):")
        for r, items in state.contents.items():
            if items:
                parts.append(f"  [{r}] {', '.join(items)}")
            else:
                parts.append(f"  [{r}] (empty)")
        parts.append("")

    # Per-object state effects (clean/hot/cool/sliced/on) acquired during rollout.
    if any(state.obj_state.values()):
        parts.append("Object state modifiers acquired earlier in this rollout:")
        for obj, sts in state.obj_state.items():
            if sts:
                parts.append(f"- {obj}: {', '.join(sorted(sts))}")
        parts.append("")

    # =============================================================
    # ## TASK CONTEXT
    # =============================================================
    parts.append("## TASK CONTEXT")
    parts.append(
        "Predict ONLY the next textworld feedback string for the action below. The agent is "
        "interacting with the env one action at a time; you are predicting what textworld "
        "would print as the immediate response to the action — exactly one short paragraph of "
        "plain English. Do NOT plan, do NOT solve the task, do NOT echo the action, do NOT "
        "list admissible commands, do NOT output JSON or markdown."
    )
    parts.append(f"Active action (verbatim, lowercase): {action_text}")
    parts.append("")

    # =============================================================
    # ## DOMAIN RULES — verbatim response templates per verb + every edge case.
    # =============================================================
    parts.append("## DOMAIN RULES")
    parts.append("Response must match the template VERBATIM. Always include instance numbers "
                 "(e.g. 'stoveburner 1' not 'stoveburner', 'pot 1' not 'pot').")
    parts.append("")
    parts.append("Example: `go to stoveburner 1` → \"You arrive at stoveburner 1. On the stoveburner 1, you see a pot 1, and a pan 2.\"")
    parts.append("")

    parts.append("- `go to <recep>`")
    parts.append("    CLOSED: \"You arrive at <recep>. The <recep> is closed.\"")
    parts.append("    OPEN+items: \"You arrive at <recep>. On the <recep>, you see a X 1, a Y 2, and a Z 1.\"")
    parts.append("    OPEN+empty: \"You arrive at <recep>. On the <recep>, you see nothing.\"")
    parts.append("    Unknown: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `open <recep>`")
    parts.append("    * Only valid when <recep> is currently CLOSED *and* agent is at <recep>.")
    parts.append("    * If valid AND <recep> contains items:")
    parts.append("        \"You open the <recep>. The <recep> is open. In it, you see <listed_items>.\"")
    parts.append("    * If valid AND <recep> is empty:")
    parts.append("        \"You open the <recep>. The <recep> is open. In it, you see nothing.\"")
    parts.append("    * Otherwise (already open, not openable, or not at it): \"Nothing happens.\"")
    parts.append("")

    parts.append("- `close <recep>`")
    parts.append("    * Only valid when <recep> is currently OPEN (and was openable in the first "
                 "place) and agent is at <recep>.")
    parts.append("    * Success: \"You close the <recep>.\"")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `take <obj> from <src>`")
    parts.append("    * Valid iff agent is at <src>, <src> is OPEN, and <obj> is currently in <src>.")
    parts.append("    * Success: \"You pick up the <obj> from the <src>.\"")
    parts.append("    * <obj> MUST include the instance number: \"You pick up the pot 1\" not \"You pick up the pot\".")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("    * After this, <obj> moves from <src> to inventory.")
    parts.append("")

    parts.append("- `put <obj> in/on <dst>` and `move <obj> to <dst>` (interchangeable)")
    parts.append("    * Valid iff <obj> is in inventory, agent is at <dst>, <dst> is OPEN.")
    parts.append("    * Success: \"You move the <obj> to the <dst>.\"  (always 'move ... to', "
                 "never 'put ... in/on', regardless of which form the agent typed.)")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `clean <obj> with <recep>` (recep must be a sinkbasin)")
    parts.append("    * Valid iff <obj> is in inventory, agent is at <recep>, <recep> is a sinkbasin.")
    parts.append("    * Success: \"You clean the <obj> using the <recep>.\"  Object gains 'clean'.")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `heat <obj> with <recep>` (recep must be a microwave)")
    parts.append("    * Valid iff <obj> is in inventory, agent is at <recep>, <recep> is a microwave.")
    parts.append("    * Microwaves do NOT need to be opened to heat — heat is a permitted "
                 "action even when the microwave is closed.")
    parts.append("    * Success: \"You heat the <obj> using the <recep>.\"  Object gains 'hot'.")
    parts.append("    * If the agent then heats again or cools, the latest action wins (sticky).")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `cool <obj> with <recep>` (recep must be a fridge)")
    parts.append("    * Valid iff <obj> is in inventory, agent is at <recep>, <recep> is a fridge.")
    parts.append("    * Fridges do NOT need to be opened to cool.")
    parts.append("    * Success: \"You cool the <obj> using the <recep>.\"  Object gains 'cool'.")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `slice <obj> with <tool>` (tool must be a knife)")
    parts.append("    * Valid iff <tool> is in inventory and <obj> is at the agent's location "
                 "or in inventory.")
    parts.append("    * Success: \"You slice the <obj> using the <tool>.\"  Object gains 'sliced'.")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `use <obj>` (typically `use desklamp N`)")
    parts.append("    * Valid iff <obj> is at the agent's location or in inventory. <obj> is an "
                 "object (not a receptacle) — desklamps, floorlamps, etc. are not listed in the "
                 "receptacle table.")
    parts.append("    * Success: \"You turn on the <obj>.\"  Object gains 'on'.")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `examine <thing>`  — read-only; never changes the world.")
    parts.append("    * If <thing> is a receptacle at the agent's location:")
    parts.append("        \"On the <thing>, you see <listed_items>.\" (or '...you see nothing.')")
    parts.append("    * If <thing> is an object in inventory or visible at current location:")
    parts.append("        \"This is a <state_words> <thing>.\"  where <state_words> reflects "
                 "any modifiers (cold, hot, clean, sliced) — e.g. \"This is a cold tomato 1.\" "
                 "If no modifier applies: \"There's nothing special about <thing>.\"")
    parts.append("    * Otherwise: \"Nothing happens.\"")
    parts.append("")

    parts.append("- `look`")
    parts.append("    * If agent is in the middle of the room (not at any receptacle):")
    parts.append("        \"You are in the middle of a room. Looking quickly around you, you "
                 "see nothing.\"  (Note: textworld emits the receptacle list ONLY in the reset "
                 "welcome message, never on a subsequent `look`.)")
    parts.append("    * If agent is facing a receptacle:")
    parts.append("        \"You are facing the <recep>. Next to it, you see <adjacent_items_or_nothing>.\"")
    parts.append("        Adjacent receptacles in the same fixture group may also be listed: "
                 "\"You are facing the shelf 2, and shelf 1. Next to it, you see ...\".")
    parts.append("")

    parts.append("- `inventory`")
    parts.append("    * If carrying nothing: \"You are not carrying anything.\"")
    parts.append("    * Otherwise: \"You are carrying: <listed_items>.\" using the same Oxford-"
                 "comma format as receptacle contents.")
    parts.append("")

    parts.append("- `help`  — emits the verbatim TextWorld command-list block (rare; agent "
                 "should not use this).")
    parts.append("")

    parts.append("Universal fallback: any action whose preconditions are not satisfied — "
                 "wrong location, missing object, closed destination for put/move, unknown "
                 "object id, unknown receptacle id, malformed verb — produces exactly:")
    parts.append("    \"Nothing happens.\"")
    parts.append("(Two words, capital N, period at end. No additional context.)")
    parts.append("")

    parts.append("Closedness rules (which classes default to closed):")
    parts.append("- Default-closed: cabinet, drawer, fridge, microwave, safe, box. Must be opened "
                 "before take/put/move using their interior. Heat/cool/clean do NOT require open.")
    parts.append("- Default-open / not-openable: countertop, shelf, sidetable, diningtable, "
                 "desk, dresser, bed, sinkbasin, stoveburner, toaster, coffeemachine, toilet, "
                 "bathtub, sofa, armchair, garbagecan, ottoman, tvstand, hand-/towel-holder, "
                 "toiletpaperhanger.")
    parts.append("- Use the receptacle table above as the authoritative open/closed source for "
                 "this specific game; it overrides the class defaults when stated.")
    parts.append("")

    # =============================================================
    # ## STEERING DIRECTIVES
    # =============================================================
    parts.append("## STEERING DIRECTIVES")
    parts.append("- Copy receptacle and object names EXACTLY from Environment State (e.g. \"desk 1\" not \"table\", \"sinkbasin 1\" not \"sink\").")
    parts.append("- Every name must include its instance number.")
    parts.append("- Only use objects/receptacles from the Environment State. Do not invent.")
    parts.append("- If uncertain, output \"Nothing happens.\"")
    parts.append("- One short paragraph. No markdown, JSON, or admissible-commands lists.")
    parts.append("- Prefix each item with \"a \" in listings (e.g. \"a apple 1, a mug 2, and a pan 1\").")
    parts.append("")

    # =============================================================
    # PREDICTION TARGET
    # =============================================================
    parts.append(
        f"PREDICTION TARGET: The single textworld feedback paragraph that the engine would "
        f"print in response to the action `{action_text}` from the environment state above."
    )
    return "\n".join(parts)


# -- Goal completion check (used by reward) ---------------------------------------

def check_goal_satisfied(meta: Dict[str, Any], messages: List[Dict[str, Any]]) -> bool:
    """Decide whether the agent's trajectory satisfies the task goal predicates.

    Reads only `messages` (assistant actions + user observations) — never touches
    the real env. Returns True iff the goal predicates are met.
    """
    if not meta.get("target_obj"):
        return False
    # `destination` is empty for look_at_obj_in_light; keep going.
    target_cls = meta["target_obj"]
    dest_cls   = meta.get("destination", "")

    cleaned: set[str] = set()
    heated:  set[str] = set()
    cooled:  set[str] = set()
    placed:  set[str] = set()        # object instance -> destination instance class
    placed_pairs: set[Tuple[str, str]] = set()  # (obj_instance, recep_instance)
    toggle_on: set[str] = set()
    held_when_toggled: set[str] = set()
    last_inventory: set[str] = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        # Track inventory changes
        for m in _RESP_TAKE.finditer(content):
            obj = m.group(1)
            if obj.split()[0].lower() == target_cls:
                last_inventory.add(obj)
        for m in _RESP_PUT.finditer(content):
            obj, dst = m.group(1), m.group(2)
            last_inventory.discard(obj)
            placed_pairs.add((obj, dst))
            if obj.split()[0].lower() == target_cls and dst.split()[0].lower() == dest_cls:
                placed.add(obj)
        for m in _RESP_CLEAN.finditer(content): cleaned.add(m.group(1))
        for m in _RESP_HEAT.finditer(content):  heated.add(m.group(1))
        for m in _RESP_COOL.finditer(content):  cooled.add(m.group(1))
        for m in _RESP_USE.finditer(content):
            tgt = m.group(1)
            toggle_on.add(tgt)
            # any held target_cls instance counts as "looked at in light"
            for held in last_inventory:
                if held.split()[0].lower() == target_cls and meta.get("toggle_target", "") in tgt:
                    held_when_toggled.add(held)

    # Per task type
    tt = meta.get("task_type", "")
    if tt == "look_at_obj_in_light":
        return bool(held_when_toggled and any(t for t in toggle_on if meta["toggle_target"] in t))
    if tt == "pick_two_obj_and_place":
        good = {o for o in placed if o.split()[0].lower() == target_cls}
        return len(good) >= 2
    if tt == "pick_clean_then_place_in_recep":
        return any(o in cleaned and o in placed for o in placed)
    if tt == "pick_heat_then_place_in_recep":
        return any(o in heated and o in placed for o in placed)
    if tt == "pick_cool_then_place_in_recep":
        return any(o in cooled and o in placed for o in placed)
    if tt == "pick_and_place_simple":
        return bool(placed)
    # fallback: any placement of target into destination
    return bool(placed)


# -- WM call payload (matches wm_proxy expectations) ------------------------------

def wm_payload(system_prompt: str, messages: List[Dict[str, Any]], action_text: str,
               max_tokens: int = 256) -> Dict[str, Any]:
    """Build the JSON body sent to wm_proxy / RunPod."""
    state_messages = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool")]
    return {
        "input": {
            "system_prompt": build_alfworld_wm_prompt(system_prompt, messages, action_text),
            "state": json.dumps(state_messages, ensure_ascii=False),
            "action": action_text,
            "max_tokens": max_tokens,
            "num_diffusion_steps": 1,
        }
    }
