"""Shared AppWorld agent prompt helpers.

Keep this prompt intentionally free of JSON schema values such as
{"type": "string"}. Those values are useful for validators, but harmful in
model-facing text because the model can copy them as real arguments.
"""

from fix_tool_names_and_schemas import TOOL_DEFS


CANONICAL_TOOL_FORMAT = (
    '[{"name": "tool_name", "parameters": {"param": "value"}}]'
)


AGENT_SYSTEM_PROMPT = """You are an AI assistant that completes tasks across multiple apps.
Always respond with exactly one tool call and no prose.

Required workflow:
1. Call supervisor__show_account_passwords to get credentials.
2. Log in to each required app using the real credentials from step 1.
3. Perform the task by calling the appropriate tools with concrete argument values.
4. Finish every task by calling supervisor__complete_task.

Output format:
- Use exactly this JSON array format:
  [{"name": "tool_name", "parameters": {"param": "value"}}]
- Do not wrap the JSON in markdown, XML tags, or commentary.
- Use the key "parameters", not "arguments".
- Make one tool call per turn.
- Do not pass access_token; tokens are managed automatically.
- Never pass schema/type placeholders as values.

Pagination:
- For list/search endpoints, use page_limit=20 unless a smaller page is needed.
- If a task requires all matching records, increment page_index until the response is empty.
- Prefer filter parameters before paginating when filters are available."""


def compact_tool_inventory() -> str:
    """Return a compact text inventory with names and parameter names only."""
    lines = ["Available tools:"]
    for tool in TOOL_DEFS:
        params = tool.get("parameters") or {}
        param_text = ", ".join(params.keys()) if params else "no parameters"
        lines.append(f"- {tool['name']}({param_text})")
    return "\n".join(lines)


def build_agent_system_prompt() -> str:
    return AGENT_SYSTEM_PROMPT + "\n\n" + compact_tool_inventory()
