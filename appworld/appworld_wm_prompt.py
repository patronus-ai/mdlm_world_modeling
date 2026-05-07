"""Action-local prompt builder for the AppWorld world model.

The SDAR WM was trained mostly on next-state prediction prompts that describe
the current action, relevant records, domain rules, and a prediction target.
This module reshapes AppWorld prompts into that style before each WM call.
"""

from __future__ import annotations

import json
import re
import ast
from typing import Any

from fix_tool_names_and_schemas import TOOL_DEFS


VALID_TOOL_NAMES = {tool["name"] for tool in TOOL_DEFS}

SECTION_BY_TOOL = {
    "spotify__show_song_library": "songs",
    "spotify__show_liked_songs": "songs",
    "spotify__search_songs": "songs",
    "spotify__show_album_library": "albums",
    "spotify__show_album": "albums",
    "spotify__show_playlist_library": "playlists",
    "spotify__show_playlist": "playlists",
    "spotify__show_recommendations": "songs",
    "spotify__show_following_artists": "following_artists",
    "spotify__show_artist": "artists",
    "venmo__show_transactions": "transactions",
    "venmo__show_received_payment_requests": "payment_requests",
    "venmo__show_sent_payment_requests": "payment_requests",
    "venmo__show_social_feed": "transactions",
    "venmo__search_friends": "venmo_friends",
    "file_system__show_directory": "directories",
    "file_system__show_file": "files",
    "simple_note__search_notes": "notes",
    "simple_note__show_note": "notes",
}

RETURN_KEY_BY_SECTION = {
    "songs": "songs",
    "albums": "albums",
    "playlists": "playlists",
    "artists": "artists",
    "following_artists": "artists",
    "transactions": "transactions",
    "payment_requests": "payment_requests",
    "venmo_friends": "friends",
    "directories": "entries",
    "files": "files",
    "notes": "notes",
}


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _section_text(prompt: str, section: str) -> str:
    match = re.search(rf"^  {re.escape(section)} .*?(?=^  [a-zA-Z_]+ \(|\Z)", prompt, re.DOTALL | re.MULTILINE)
    return match.group(0).rstrip() if match else ""


def _section_total(section_text: str) -> int:
    match = re.search(r"\((\d+) total", section_text)
    return int(match.group(1)) if match else 0


def _record_lines(section_text: str) -> list[str]:
    lines = []
    current = []
    for line in section_text.splitlines()[1:]:
        if line.startswith("    - "):
            if current:
                lines.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        lines.append("\n".join(current))
    return lines


def _extract_credentials(prompt: str) -> dict[str, tuple[str, str]]:
    creds = {}
    for match in re.finditer(r"- ([\w_]+): username=([^,]+), password=(\S+)", prompt):
        creds[match.group(1)] = (match.group(2), match.group(3))
    return creds


def _extract_field(record: str, field: str) -> str | None:
    pattern = rf"\b{re.escape(field)}: (.*?)(?=, [a-zA-Z_]+:|$|\n)"
    match = re.search(pattern, record, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_scalar(value: str) -> Any:
    value = value.strip().rstrip(",")
    if value in {"True", "False"}:
        return value == "True"
    if value in {"None", "null"}:
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def _record_to_object(record: str) -> dict[str, Any]:
    text = record.strip()
    if text.startswith("- "):
        text = text[2:]
    starts = list(re.finditer(r"(?:(?<=, )|^)([a-zA-Z_]+): ", text))
    obj: dict[str, Any] = {}
    for index, match in enumerate(starts):
        key = match.group(1)
        value_start = match.end()
        value_end = starts[index + 1].start() - 2 if index + 1 < len(starts) else len(text)
        value = text[value_start:value_end].strip()
        obj[key] = _parse_scalar(value)
    return obj


def _records_to_objects(records: list[str]) -> list[dict[str, Any]]:
    return [_record_to_object(record) for record in records]


def _extract_int(record: str, *fields: str) -> int | None:
    for field in fields:
        value = _extract_field(record, field)
        if value is None:
            continue
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return None


def _norm(value: Any) -> str:
    return str(value).strip().strip('"').strip("'").lower()


def _schema_arg_present(args: dict[str, Any]) -> bool:
    for value in args.values():
        if isinstance(value, dict) and "type" in value:
            return True
        if isinstance(value, str) and ("{'type'" in value or '{"type"' in value):
            return True
    return False


def _page_args(args: dict[str, Any]) -> tuple[int, int]:
    try:
        page_index = max(0, _safe_int(args.get("page_index", 0)))
    except Exception:
        page_index = 0
    try:
        page_limit = min(20, max(1, _safe_int(args.get("page_limit", 5), 5)))
    except Exception:
        page_limit = 5
    return page_index, page_limit


def _record_matches(tool_name: str, args: dict[str, Any], record: str) -> bool:
    if tool_name == "spotify__show_liked_songs" and "liked: True" not in record:
        return False
    if tool_name == "spotify__search_songs":
        genre = args.get("genre")
        if genre and _norm(_extract_field(record, "genre")) != _norm(genre):
            return False
        album_id = args.get("album_id")
        if album_id is not None and _extract_int(record, "album_id") != _safe_int(album_id):
            return False
        artist_id = args.get("artist_id")
        if artist_id is not None and _extract_int(record, "artist_id") != _safe_int(artist_id):
            return False
        query = args.get("query")
        if query and _norm(query) not in _norm(record):
            return False
        for arg_name, field in [
            ("min_play_count", "play_count"),
            ("max_play_count", "play_count"),
            ("min_like_count", "like_count"),
            ("max_like_count", "like_count"),
        ]:
            if arg_name not in args:
                continue
            value = _extract_int(record, field)
            if value is None:
                return False
            threshold = _safe_int(args[arg_name])
            if arg_name.startswith("min_") and value < threshold:
                return False
            if arg_name.startswith("max_") and value > threshold:
                return False
        min_date = args.get("min_release_date")
        max_date = args.get("max_release_date")
        release_date = _extract_field(record, "release_date")
        if min_date and release_date and release_date < str(min_date):
            return False
        if max_date and release_date and release_date > str(max_date):
            return False
    if tool_name == "venmo__show_transactions":
        direction = args.get("direction")
        if direction and _norm(_extract_field(record, "direction")) != _norm(direction):
            return False
        query = args.get("query")
        if query and _norm(query) not in _norm(record):
            return False
        for arg_name, field in [("min_amount", "amount"), ("max_amount", "amount"), ("min_like_count", "like_count")]:
            if arg_name not in args:
                continue
            value = _extract_field(record, field)
            if value is None:
                return False
            number = float(re.search(r"-?\d+(?:\.\d+)?", value).group(0))
            threshold = float(args[arg_name])
            if arg_name.startswith("min_") and number < threshold:
                return False
            if arg_name.startswith("max_") and number > threshold:
                return False
        for arg_name, field in [("min_created_at", "created_at"), ("max_created_at", "created_at")]:
            if arg_name not in args:
                continue
            value = _extract_field(record, field)
            if value is None:
                return False
            if arg_name.startswith("min_") and value < str(args[arg_name]):
                return False
            if arg_name.startswith("max_") and value > str(args[arg_name]):
                return False
    if tool_name in {"venmo__show_received_payment_requests", "venmo__show_sent_payment_requests"}:
        expected_direction = "received" if "received" in tool_name else "sent"
        if _norm(_extract_field(record, "direction")) != expected_direction:
            return False
        status = args.get("status")
        if status:
            status_l = _norm(status)
            if status_l == "pending" and ("approved_at:" in record or "denied_at:" in record):
                return False
            if status_l == "approved" and "approved_at:" not in record:
                return False
            if status_l == "denied" and "denied_at:" not in record:
                return False
    if tool_name == "file_system__show_directory":
        path = args.get("directory_path") or args.get("path")
        if path and _norm(path) not in _norm(record):
            return False
        substring = args.get("substring")
        if substring and _norm(substring) not in _norm(record):
            return False
        entry_type = _norm(args.get("entry_type", ""))
        if entry_type == "files":
            return False
        if entry_type == "directories" and "tilde_path:" not in record:
            return False
    if tool_name == "simple_note__search_notes":
        query = args.get("query")
        if query:
            obj = _record_to_object(record)
            searchable = " ".join(str(obj.get(key, "")) for key in ("title", "content", "tags"))
            if _norm(query) not in _norm(searchable):
                return False
    return True


def _target_record(tool_name: str, args: dict[str, Any], records: list[str]) -> list[str]:
    id_args = {
        "spotify__show_album": ("album_id", "id"),
        "spotify__show_playlist": ("playlist_id", "id"),
        "spotify__show_artist": ("artist_id", "id"),
        "simple_note__show_note": ("note_id", "id"),
        "file_system__show_file": ("file_path", "tilde_path"),
    }
    if tool_name not in id_args:
        return records
    arg_name, field = id_args[tool_name]
    expected = args.get(arg_name)
    if expected is None:
        return []
    matched = []
    for record in records:
        if field == "tilde_path":
            if _norm(expected) == _norm(_extract_field(record, field)):
                matched.append(record)
        else:
            try:
                if _extract_int(record, field) == int(expected):
                    matched.append(record)
            except (ValueError, TypeError):
                pass
    return matched


def _record_context(tool_name: str, args: dict[str, Any], section: str, prompt: str) -> tuple[int, list[str], list[str]]:
    text = _section_text(prompt, section)
    records = _record_lines(text)
    if tool_name in {"spotify__show_album", "spotify__show_playlist", "spotify__show_artist", "simple_note__show_note", "file_system__show_file"}:
        matched = _target_record(tool_name, args, records)
        return len(matched), matched[:5], matched
    matched = [record for record in records if _record_matches(tool_name, args, record)]
    page_index, page_limit = _page_args(args)
    start = page_index * page_limit
    page = matched[start:start + page_limit]
    return len(matched), page, matched


def _tool_signature(tool_name: str) -> str:
    for tool in TOOL_DEFS:
        if tool["name"] == tool_name:
            params = ", ".join((tool.get("parameters") or {}).keys()) or "no parameters"
            return f"{tool_name}({params})"
    return f"{tool_name}(unknown)"


def _validation_error(system_prompt: str, tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name not in VALID_TOOL_NAMES:
        return f"Unknown tool: {tool_name}"
    if _schema_arg_present(args):
        return "Invalid argument values: schema/type dictionaries are not runtime values"
    if tool_name.endswith("__login"):
        app = tool_name.split("__", 1)[0]
        creds = _extract_credentials(system_prompt)
        if app in creds:
            username, password = creds[app]
            if args.get("username") != username or args.get("password") != password:
                return "Invalid username or password"
    section = SECTION_BY_TOOL.get(tool_name)
    if section and tool_name in {"spotify__show_album", "spotify__show_playlist", "spotify__show_artist", "simple_note__show_note", "file_system__show_file"}:
        _, page, _ = _record_context(tool_name, args, section, system_prompt)
        if not page:
            return "Record not found"
    return None


def expected_appworld_response(system_prompt: str, tool_name: str, tool_args: dict[str, Any],
                               logged_in_apps: set[str] | None = None) -> str | None:
    """Return deterministic JSON for AppWorld calls we can validate locally.

    Args:
        logged_in_apps: set of app names that have been successfully logged in.
            If provided, non-login API calls to apps not in this set return 401.
    """
    validation_error = _validation_error(system_prompt, tool_name, tool_args)
    if validation_error:
        return json.dumps({"error": validation_error}, ensure_ascii=False)

    # Auth enforcement: require login before calling app APIs
    # Exempt: supervisor APIs, login itself, and search_songs (works without auth in real AppWorld)
    if logged_in_apps is not None and "__" in tool_name:
        app = tool_name.split("__", 1)[0]
        api = tool_name.split("__", 1)[1]
        exempt = (app == "supervisor" or api == "login" or tool_name == "spotify__search_songs")
        if not exempt and app not in logged_in_apps:
            return json.dumps({"error": "401 Unauthorized. Please login first."}, ensure_ascii=False)

    section = SECTION_BY_TOOL.get(tool_name)
    if section:
        matched_count, page_records, _ = _record_context(tool_name, tool_args, section, system_prompt)
        objects = _records_to_objects(page_records)
        if tool_name in {"spotify__show_album", "spotify__show_playlist", "spotify__show_artist", "simple_note__show_note", "file_system__show_file"}:
            if not objects:
                return json.dumps({"error": "Record not found"}, ensure_ascii=False)
            return json.dumps(objects[0], ensure_ascii=False)
        return json.dumps({
            "total": matched_count,
            RETURN_KEY_BY_SECTION.get(section, "data"): objects,
        }, ensure_ascii=False)

    if tool_name.endswith("__login"):
        return json.dumps({"access_token": "tok_valid", "token_type": "Bearer"}, ensure_ascii=False)

    mutation_prefixes = (
        "spotify__like_", "spotify__unlike_", "spotify__review_", "spotify__create_",
        "spotify__add_", "spotify__remove_", "spotify__follow_", "spotify__download_",
        "venmo__create_", "venmo__approve_", "venmo__like_", "venmo__remind_", "venmo__deny_",
        "phone__send_", "phone__delete_",
        "file_system__create_", "file_system__move_", "file_system__delete_", "file_system__compress_",
        "simple_note__update_", "simple_note__create_", "simple_note__delete_",
    )
    if tool_name.startswith(mutation_prefixes):
        return json.dumps({"status": "success", "message": "Action completed"}, ensure_ascii=False)

    return None


def build_action_local_wm_prompt(system_prompt: str, state: list[Any], tool_name: str, tool_args: dict[str, Any]) -> str:
    """Build a training-style prompt for a single AppWorld WM action."""
    validation_error = _validation_error(system_prompt, tool_name, tool_args)
    section = SECTION_BY_TOOL.get(tool_name)
    total = 0
    page_records: list[str] = []
    matched_count = 0
    if section and validation_error is None:
        matched_count, page_records, _ = _record_context(tool_name, tool_args, section, system_prompt)
        total = matched_count

    creds_block = "\n".join(re.findall(r"- [\w_]+: username=[^,]+, password=\S+", system_prompt))
    active_task = re.search(r"Active task: (.*?)(?:\n|$)", system_prompt)
    current_date = re.search(r"Current date: (.*?)(?:\n|$)", system_prompt)
    return_key = RETURN_KEY_BY_SECTION.get(section or "", "data")
    page_index, page_limit = _page_args(tool_args)

    parts = [
        "## ENVIRONMENT STATE",
        "The environment is AppWorld tool-response prediction.",
        "",
    ]
    if active_task:
        parts.extend(["Active task:", f"- {active_task.group(1)}", ""])
    if current_date:
        parts.extend(["Current date:", f"- {current_date.group(1)}", ""])
    if creds_block:
        parts.extend(["Account credentials:", creds_block, ""])

    validation_status = "invalid" if validation_error else "valid"
    parts.extend([
        "Active tool call:",
        f"- tool_name: {tool_name}",
        f"- arguments: {json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}",
        f"- validation_status: {validation_status}",
        f"- validation_error: {validation_error or 'null'}",
        "",
        "## TASK CONTEXT",
        "Predict only the JSON tool response for this one AppWorld action. Do not solve the whole user task. Do not echo the action.",
        "",
        "## TOOL SCHEMAS",
        f"- { _tool_signature(tool_name) }",
    ])

    if section:
        parts.extend([
            f"- Return shape for list/search tools: {{\"total\": int, \"{return_key}\": array}}",
            f"- Relevant source section: {section}",
        ])
    parts.extend([
        "- For invalid calls, return exactly one JSON object with an error key.",
        "",
        "## DOMAIN RULES",
        "- Validate the tool name and argument value types before generating a response.",
        "- If validation_status is valid, the active tool is available and the arguments passed validation.",
    ])
    if validation_error:
        parts.append("- Unknown tools, bad credentials, schema/type placeholder arguments, and missing target records return an error JSON object.")
    else:
        parts.append("- This is not an unknown-tool case. Do not return an error for tool availability.")
    parts.extend([
        "- Returned records and IDs must come only from the record lines shown below.",
        "- Return strict valid JSON only. Use JSON true/false/null, not Python True/False/None.",
        "- No markdown, no commentary, no tracebacks, and do not double-quote or escape the whole JSON object.",
        "- Do not include <think> text. Wrap the final JSON in <tool_response> only if the model requires tags.",
    ])

    if section and validation_error is None:
        page_objects = _records_to_objects(page_records)
        parts.extend([
            "",
            "## COMPUTED QUERY STATE",
            f"- matched_count_after_filters: {matched_count}",
            f"- page_index: {page_index}",
            f"- page_limit: {page_limit}",
            f"- page_offset: {page_index * page_limit}",
            f"- records_on_this_page: {len(page_records)}",
            "- The response total field must equal matched_count_after_filters.",
            "- The response array must contain only the records_on_this_page listed below.",
            "",
            "Strict JSON records for this response page:",
        ])
        if page_objects:
            parts.append(json.dumps(page_objects, ensure_ascii=False, indent=2))
        else:
            parts.append("[]")

    parts.extend([
        "",
        "## STEERING DIRECTIVES",
        f"- Expected validation status: {validation_status}.",
    ])
    if validation_error:
        parts.append(f"- The exact error response body is: {{\"error\": {json.dumps(validation_error)}}}.")
    else:
        parts.append("- The tool call is valid; do not return an error.")
    parts.extend([
        "- For list/search tools, apply filters before pagination; never repeat page 0 for later pages.",
        "- For login success, return a short access token only when credentials match Account credentials.",
        f"- If this is a list/search success, return {{\"total\": {total}, \"{return_key}\": [...]}} using only Strict JSON records for this response page.",
        "",
        "PREDICTION TARGET: The strict JSON tool response for the action in the user turn.",
    ])
    return "\n".join(parts)
