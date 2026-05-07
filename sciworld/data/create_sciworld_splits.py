#!/usr/bin/env python3
"""Generate RL and test splits for ScienceWorld tasks.
v2: Includes full room-by-room contents in WM system prompt."""
import json, random, sys
sys.path.insert(0, "/workspace/user/sciworld_training")

random.seed(42)

SYSTEM_PROMPT = """You are an agent in ScienceWorld, a text-based science experiment environment. You navigate rooms, pick up objects, and perform experiments to complete science tasks.

At each turn, output EXACTLY ONE action as plain text. Do not wrap it in JSON, markdown, or any other format. Just output the action text.

Common actions:
- look around (see what's in the current room)
- go to {location} (move to a room)
- pick up {object} (take an object)
- put {object} in {container} (place object somewhere)
- open {object} / close {object}
- activate {object} / deactivate {object} (turn things on/off)
- pour {object} in {object}
- use {object} on {object}
- focus on {object} (direct attention to a substance/object — WARNING: this submits your answer. Focusing on the wrong thing = instant failure with score -100!)
- mix {object}
- wait / wait1
- inventory (check what you're carrying)
- teleport to {location} (instant travel)
- connect {object} to {object} (for electrical circuits)

CRITICAL RULES:
- "focus on X" is how you SUBMIT YOUR ANSWER. Only focus on the correct target after completing ALL prerequisite steps.
- For find tasks: focus on the correct object, then move it to the designated box.
- For state-change tasks (boil/melt/freeze): focus on the substance FIRST, then heat/cool it.
- For measurement tasks: use the instrument FIRST (thermometer, etc.), then determine the answer.
- For genetics tasks: the colored boxes are multiple-choice answers (e.g., blue=dominant, orange=recessive). Focus on the correct BOX, not the plant/seed.
- NEVER focus on "agent" — this always gives -100.
- You MUST "focus on" something to submit your answer. If you never focus, your score is 0. Exploring without submitting is WORSE than making an educated guess.
- Focus on your best guess after gathering enough information. Do not wait until you are 100% certain — a good guess scores more than no answer.

EXPLORATION STRATEGY — follow this approach:
1. First, READ the task description carefully. Identify what you need (substances, instruments, containers, answer boxes).
2. EXPLORE: Use "teleport to {room}" to visit rooms and "look around" to find needed objects. You MUST search multiple rooms — objects are spread across 10 locations.
3. GATHER: Pick up needed objects (containers, substances, instruments).
4. PERFORM: Do the experiment (heat, cool, mix, connect, grow, etc.).
5. ANSWER: Only THEN use "focus on" to submit your answer.

MOVEMENT: Always use "teleport to {room}" to move between rooms. "go to {room}" only works for adjacent rooms and will fail otherwise. "teleport to" ALWAYS works and is instant.

Room layout:
- hallway: central hub, connects to most rooms
- kitchen: stove, freezer, fridge, sink, cups, pots, food
- workshop: batteries, wires, light bulbs, switches, tools
- greenhouse: flower pots, seeds, jars, bee hive, soil, shovel
- foundry: blast furnace (very hot), sink
- outside: fire pit, fountain (water), axe, wood, ground, animals
- bedroom: bed, closet, table
- living room: chair, couch, desk, sometimes answer boxes
- art studio: paint cups, jug, cupboard
- bathroom: bathtub, sink, toilet, sometimes answer boxes

Object naming — USE EXACT NAMES from "look around":
- If you see "a substance called caesium", the action is "pick up caesium" (not "pick up substance")
- If you see "a glass cup (containing water)", the action is "pick up glass cup"
- If you see "a ceramic cup (containing caesium)", pick up "ceramic cup" to get the caesium
- If you see "a baby wolf", the action is "focus on baby wolf" or "pick up baby wolf"
- If "pick up X" fails, try "look around" to see the exact object names

SUBSTANCE vs CONTAINER — CRITICAL DISTINCTION:
- When a task asks about a SUBSTANCE (e.g., "boil tin", "melt caesium", "freeze water"):
  - The substance is INSIDE a container (e.g., "ceramic cup containing caesium", "tin cup containing tin")
  - You must "focus on" the SUBSTANCE NAME: "focus on caesium" or "focus on tin"
  - Do NOT focus on the container: "focus on tin cup" focuses on the CUP, not the substance tin!
  - "tin cup" = a container made of tin. "tin" = the metal substance. They are DIFFERENT.
  - "ceramic cup" = a container. "caesium" = the substance inside it.
  - Pick up the CONTAINER to carry the substance: "pick up ceramic cup" moves both to inventory
- For state-change tasks (boil/melt/freeze): FIRST focus on the substance, THEN heat/cool it
- WRONG: "focus on tin cup" (this focuses on the container, score = -100)
- CORRECT: "focus on tin" (this focuses on the substance)

ANIMAL LIFESPANS (for lifespan comparison tasks):
- Longest to shortest lived: giant tortoise > elephant > parrot > dog > wolf > cat > hamster > beaver > chipmunk > mouse > dragonfly > bee > ant
- When asked for "longest lived": choose the animal highest on this list
- When asked for "shortest lived": choose the animal lowest on this list (insects: ant, bee, dragonfly)
- Giant tortoise lives ~100+ years, beaver ~10-15 years, ant/bee ~weeks to months

SEARCHING FOR HIDDEN OBJECTS:
- Many objects are inside CLOSED containers: cupboard, closet, fridge, freezer, drawer, cabinet, cardboard box, oven
- You MUST "open {container}" before you can see or pick up objects inside
- If you can't find a substance after visiting all rooms, try: "open cupboard", "open freezer", "open fridge", "open closet", "open drawer", "open cabinet", "open oven"
- After opening, do "look around" to see the newly revealed contents

Tips:
- If you can't find an object, FIRST try opening containers in the current room, THEN teleport to other rooms
- If "go to {room}" fails, use "teleport to {room}" instead — it always works
- READ the task description to identify WHAT you need, then search rooms to find WHERE it is
- Heat sources: stove (kitchen), blast furnace (foundry), fire pit (outside)
- Cold sources: freezer (kitchen), fridge (kitchen)
- Common containers: glass cup, metal pot, bowl, jar, jug
- Animals are usually outside or in the greenhouse
- Answer boxes (colored boxes) are usually in living room, bedroom, or bathroom
- Substances for experiments may be in ANY room — always search if not visible
- For thermometer tasks: use "use thermometer on X" to measure temperature BEFORE submitting answer"""


def explore_all_rooms(env):
    """Teleport to every room and record contents."""
    rooms = {}
    room_names = ['hallway', 'kitchen', 'workshop', 'greenhouse', 'bedroom',
                  'living room', 'art studio', 'foundry', 'bathroom', 'outside']
    for room in room_names:
        env.step(f'teleport to {room}')
        obs, _, _, _ = env.step('look around')
        # Convert tab-indented format to plain text for WM compatibility
        lines = []
        for line in obs.split('\n'):
            line = line.strip().strip('\t')
            if line and line != 'the agent':
                lines.append(line)
        rooms[room] = ' '.join(lines) if lines else obs
    return rooms


def build_wm_system_prompt(task_desc, initial_obs, task_name, valid_actions, valid_objects, room_contents):
    """Build the WM system prompt with FULL environment map."""

    # Format room contents compactly
    room_block = []
    for room, obs in room_contents.items():
        # Truncate long room descriptions
        room_block.append(f"[{room}]\n{obs}")
    rooms_text = "\n\n".join(room_block)

    return f"""## ENVIRONMENT
ScienceWorld text-based interactive environment.
10 interconnected rooms. The agent navigates between rooms, picks up objects, and performs science experiments.

Task: {task_desc}
Task type: {task_name}

## STARTING STATE
{initial_obs}

## COMPLETE ROOM MAP
{rooms_text}

## ROOM CONNECTIONS
- hallway connects to: kitchen, bedroom, living room, art studio, greenhouse, workshop
- kitchen connects to: hallway, outside, bathroom
- outside connects to: kitchen, foundry, greenhouse
- foundry connects to: outside
- greenhouse connects to: hallway, outside
- workshop connects to: hallway
- bedroom connects to: hallway
- living room connects to: hallway
- art studio connects to: hallway
- bathroom connects to: kitchen

## ACTION RESPONSES
When the agent does "teleport to X": respond "You teleport to the X." (always succeeds for any room)
When the agent does "go to X": ONLY succeeds if X is adjacent (see ROOM CONNECTIONS above). If adjacent, respond "You move to the X." If NOT adjacent, respond "No known action matches that input."
When the agent does "look around": respond with the EXACT room contents from the COMPLETE ROOM MAP above for the agent's CURRENT room.
When the agent does "pick up X": respond "You move the X to the inventory." (only if X is in current room)
When the agent does "activate X": respond "The X is now activated." (only if X is in current room)
When the agent does "deactivate X": respond "The X is now deactivated."
When the agent does "focus on X": respond "You focus on the X." (only if X is in current room — otherwise "No known action matches that input.")
When the agent does "put X in/on Y": respond "You move the X to the Y."
When the agent does "open X": respond "You open the X." or "The X is already open."
When the agent does "inventory": list items the agent has picked up.
When the agent does "wait" or "wait1": respond "(1 tick passes)"
When action is invalid or object not present: respond "No known action matches that input."

## IMPORTANT RULES
- Objects can ONLY be interacted with if the agent is in the SAME room as the object.
- "go to X" ONLY works for adjacent rooms. Use ROOM CONNECTIONS to check adjacency. "teleport to X" works for ANY room.
- "focus on X" fails if X is not in the current room. Return "No known action matches that input."
- "pick up X" fails if X is not in the current room. Return "No known action matches that input."
- Use ONLY objects that appear in the COMPLETE ROOM MAP above. Do NOT invent objects.
- After "pick up X", the object moves from the room to inventory. It is no longer in the room.
- After "put X in Y", X moves from inventory to Y.

Available action templates: {', '.join(valid_actions[:15])}"""


def main():
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("")
    task_names = env.get_task_names()

    rl_tasks = []
    test_tasks = []

    for task_name in task_names:
        env.load(task_name, 0, "easy")
        max_var = env.get_max_variations(task_name)

        train_vars = set()
        test_vars = set()

        for _ in range(10):
            tv = env.get_random_variation_train()
            if tv not in train_vars and len(train_vars) < 2:
                train_vars.add(tv)
            tv2 = env.get_random_variation_test()
            if tv2 not in test_vars and tv2 not in train_vars and len(test_vars) < 2:
                test_vars.add(tv2)

        for v in range(max_var):
            if len(train_vars) >= 2 and len(test_vars) >= 2:
                break
            if v not in train_vars and v not in test_vars:
                if len(train_vars) < 2:
                    train_vars.add(v)
                elif len(test_vars) < 2:
                    test_vars.add(v)

        for split_name, var_set, task_list in [("train", train_vars, rl_tasks), ("test", test_vars, test_tasks)]:
            for var_idx in var_set:
                env.load(task_name, var_idx, "easy")
                obs, _ = env.reset()
                task_desc = env.get_task_description()
                valid_actions = env.get_possible_actions()
                valid_objects = env.get_possible_objects()

                # Explore all rooms for this variation
                room_contents = explore_all_rooms(env)

                # Reset to get back to starting position
                env.load(task_name, var_idx, "easy")
                obs, _ = env.reset()

                task_id = f"{task_name}_v{var_idx}"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Task: {task_desc}\n\nYou are in: {obs}"},
                ]
                wm_prompt = build_wm_system_prompt(
                    task_desc, obs, task_name, valid_actions, valid_objects, room_contents
                )

                task_list.append({
                    "messages": json.dumps(messages),
                    "task_id": task_id,
                    "instruction": task_desc,
                    "ground_truth": "",
                    "wm_system_prompt": wm_prompt,
                    "max_turns": 50,
                })

    env.close()

    with open("data/sciworld_rl_split.jsonl", "w") as f:
        for t in rl_tasks:
            f.write(json.dumps(t) + "\n")

    with open("data/sciworld_test_split.jsonl", "w") as f:
        for t in test_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Created {len(rl_tasks)} RL tasks and {len(test_tasks)} test tasks")
    # Check WM prompt sizes
    sizes = [len(t['wm_system_prompt']) for t in rl_tasks + test_tasks]
    print(f"WM prompt sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

if __name__ == "__main__":
    main()
