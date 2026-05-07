#!/usr/bin/env python3
"""Generate RL and test splits for ScienceWorld — LFM-optimized.
Shorter system prompt + few-shot examples for 1.2B model."""
import json, random, sys
sys.path.insert(0, "/workspace/user/sciworld_training")

random.seed(42)

SYSTEM_PROMPT = """You are a ScienceWorld agent. Output EXACTLY ONE action per turn as plain text. No explanations, no sentences, no markdown — just the action command.

Valid actions: look around, teleport to {room}, pick up {object}, put {object} in {container}, open {object}, close {object}, activate {object}, deactivate {object}, focus on {object}, pour {object} in {object}, use {object} on {object}, mix {object}, connect {object} to {object}, wait, inventory

Rooms: hallway, kitchen, workshop, greenhouse, bedroom, living room, art studio, foundry, bathroom, outside

RULES:
- Always use "teleport to {room}" to move (always works). "go to" often fails.
- "focus on {object}" SUBMITS YOUR ANSWER. Wrong focus = -100 penalty. Only focus when CERTAIN.
- For substances in containers: focus on the SUBSTANCE ("focus on tin"), NOT the container ("focus on tin cup").
- Explore rooms with "teleport to" + "look around". Open containers (cupboard, freezer, fridge, closet) to find hidden objects.
- Heat: stove (kitchen), blast furnace (foundry), fire pit (outside). Cold: freezer, fridge (kitchen).
- Animal lifespans longest→shortest: giant tortoise > elephant > parrot > dog > wolf > cat > beaver > chipmunk > mouse > dragonfly > bee > ant

STRATEGY: 1) Read task. 2) Explore rooms. 3) Gather objects. 4) Do experiment. 5) Submit answer with focus."""

# Few-shot examples showing correct output format
FEW_SHOT = [
    {"role": "user", "content": "Task: Find a non-living thing.\n\nYou are in: This room is called the hallway. In it, you see: a substance called air, a painting."},
    {"role": "assistant", "content": "teleport to kitchen"},
    {"role": "user", "content": "You teleport to the kitchen."},
    {"role": "assistant", "content": "look around"},
    {"role": "user", "content": "This room is called the kitchen. In it, you see: a stove, a glass cup (containing water), a freezer, a fridge, a counter."},
    {"role": "assistant", "content": "pick up glass cup"},
    {"role": "user", "content": "You move the glass cup to the inventory."},
    {"role": "assistant", "content": "teleport to bedroom"},
    {"role": "user", "content": "You teleport to the bedroom."},
    {"role": "assistant", "content": "look around"},
    {"role": "user", "content": "This room is called the bedroom. In it, you see: a bed, a closet, a table, a green box, a blue box."},
    {"role": "assistant", "content": "focus on glass cup"},
    {"role": "user", "content": "You focus on the glass cup."},
    {"role": "assistant", "content": "put glass cup in green box"},
    {"role": "user", "content": "You move the glass cup to the green box."},
]


def explore_all_rooms(env):
    rooms = {}
    room_names = ['hallway', 'kitchen', 'workshop', 'greenhouse', 'bedroom',
                  'living room', 'art studio', 'foundry', 'bathroom', 'outside']
    for room in room_names:
        env.step(f'teleport to {room}')
        obs, _, _, _ = env.step('look around')
        lines = []
        for line in obs.split('\n'):
            line = line.strip().strip('\t')
            if line and line != 'the agent':
                lines.append(line)
        rooms[room] = ' '.join(lines) if lines else obs
    return rooms


def build_wm_system_prompt(task_desc, initial_obs, task_name, valid_actions, valid_objects, room_contents):
    room_block = []
    for room, obs in room_contents.items():
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
When the agent does "focus on X": respond "You focus on the X." (only if X is in current room or inventory — otherwise "No known action matches that input.")
When the agent does "put X in/on Y": respond "You move the X to the Y."
When the agent does "open X": respond "You open the X." or "The X is already open."
When the agent does "inventory": list items the agent has picked up.
When the agent does "wait" or "wait1": respond "(1 tick passes)"
When action is invalid or object not present: respond "No known action matches that input."

## IMPORTANT RULES
- Objects can ONLY be interacted with if the agent is in the SAME room as the object.
- "go to X" ONLY works for adjacent rooms. Use ROOM CONNECTIONS to check adjacency. "teleport to X" works for ANY room.
- "focus on X" fails if X is not in the current room or inventory. Return "No known action matches that input."
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

                room_contents = explore_all_rooms(env)

                env.load(task_name, var_idx, "easy")
                obs, _ = env.reset()

                task_id = f"{task_name}_v{var_idx}"
                # Build messages with few-shot examples
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
                messages.extend(FEW_SHOT)
                # Now the actual task
                messages.append(
                    {"role": "user", "content": f"NEW TASK: {task_desc}\n\nYou are in: {obs}"},
                )
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

    with open("data/sciworld_rl_split_lfm.jsonl", "w") as f:
        for t in rl_tasks:
            f.write(json.dumps(t) + "\n")

    with open("data/sciworld_test_split_lfm.jsonl", "w") as f:
        for t in test_tasks:
            f.write(json.dumps(t) + "\n")

    print(f"Created {len(rl_tasks)} RL tasks and {len(test_tasks)} test tasks (LFM)")
    sizes = [len(t['wm_system_prompt']) for t in rl_tasks + test_tasks]
    print(f"WM prompt sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    msg_sizes = [len(t['messages']) for t in rl_tasks + test_tasks]
    print(f"Message sizes: min={min(msg_sizes)}, max={max(msg_sizes)}, avg={sum(msg_sizes)//len(msg_sizes)}")

if __name__ == "__main__":
    main()
