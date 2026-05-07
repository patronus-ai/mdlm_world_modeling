#!/usr/bin/env python3
"""
Fix tool names and schemas across all dataset splits.

Two problems:
1. Several tool names don't match real AppWorld API names
2. Tool schemas are missing pagination and filter params

This script:
- Renames tools in system prompts (JSON tool definitions)
- Renames tools in assistant messages (tool calls in trajectories)
- Replaces the tool schema block in WM system prompts
- Updates all three splits + SFT data
"""
import json
import re
import sys

# ── Name mapping: our_name -> real AppWorld API name ──
NAME_MAP = {
    "spotify__show_playlists": "spotify__show_playlist_library",
    "spotify__show_playlist_songs": "spotify__show_playlist",
    "venmo__show_pending_requests": "venmo__show_received_payment_requests",
    "venmo__send_payment": "venmo__create_transaction",
    "venmo__accept_payment_request": "venmo__approve_payment_request",
    "venmo__add_comment": "venmo__create_transaction_comment",
    "venmo__send_reminder": "venmo__remind_payment_request",
    "phone__show_text_messages": "phone__search_text_messages",
    "phone__show_voice_messages": "phone__search_voice_messages",
    "file_system__read_file": "file_system__show_file",
    "simple_note__show_notes": "simple_note__search_notes",
}

# Reverse map for convenience
REVERSE_NAME_MAP = {v: k for k, v in NAME_MAP.items()}

# Also fix param names that differ
PARAM_RENAMES = {
    "venmo__create_transaction": {
        "recipient_username": "receiver_email",
        "note": "description",
    },
    "venmo__approve_payment_request": {
        "request_id": "payment_request_id",
    },
    "venmo__create_transaction_comment": {
        "text": "comment",
    },
    "venmo__remind_payment_request": {
        "request_id": "payment_request_id",
    },
    "phone__search_text_messages": {
        "recipient": None,  # param name stays same for send
    },
    "phone__send_text_message": {
        "recipient": "phone_number",
        "text": "message",
    },
    "file_system__show_file": {
        "file_path": "file_path",  # same name, OK
    },
}

# ── Full tool definitions with correct names and ALL params ──
TOOL_DEFS = [
    {"name": "supervisor__show_account_passwords", "description": "Show login credentials for all apps", "parameters": {}},
    {"name": "supervisor__complete_task", "description": "Mark task complete with answer or summary", "parameters": {"answer": {"type": "string"}}},

    # Spotify
    {"name": "spotify__login", "description": "Log in to Spotify", "parameters": {"username": {"type": "string"}, "password": {"type": "string"}}},
    {"name": "spotify__show_playlist_library", "description": "Show your playlists", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_playlist", "description": "Show detailed info and songs for a specific playlist", "parameters": {
        "playlist_id": {"type": "integer"},
    }},
    {"name": "spotify__search_songs", "description": "Search songs with filters", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "artist_id": {"type": "integer", "description": "Filter by artist ID"},
        "album_id": {"type": "integer", "description": "Filter by album ID"},
        "genre": {"type": "string", "description": "Filter by genre"},
        "min_release_date": {"type": "string", "description": "Min release date YYYY-MM-DD"},
        "max_release_date": {"type": "string", "description": "Max release date YYYY-MM-DD"},
        "min_play_count": {"type": "integer", "description": "Min play count"},
        "max_play_count": {"type": "integer", "description": "Max play count"},
        "min_like_count": {"type": "integer", "description": "Min like count"},
        "max_like_count": {"type": "integer", "description": "Max like count"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_liked_songs", "description": "Show songs you liked", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_song_library", "description": "Show your song library", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_album_library", "description": "Show your album library", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_album", "description": "Show album details", "parameters": {"album_id": {"type": "integer"}}},
    {"name": "spotify__show_recommendations", "description": "Show song recommendations", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_artist", "description": "Show artist details including follower_count", "parameters": {"artist_id": {"type": "integer"}}},
    {"name": "spotify__show_following_artists", "description": "Show artists you follow", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "spotify__show_current_song", "description": "Show the currently playing song", "parameters": {}},
    {"name": "spotify__next_song", "description": "Skip to next song in queue", "parameters": {}},
    {"name": "spotify__previous_song", "description": "Go to previous song in queue", "parameters": {}},
    {"name": "spotify__like_song", "description": "Like a song", "parameters": {"song_id": {"type": "integer"}}},
    {"name": "spotify__unlike_song", "description": "Unlike a song", "parameters": {"song_id": {"type": "integer"}}},
    {"name": "spotify__review_song", "description": "Review/rate a song (1-5 stars)", "parameters": {"song_id": {"type": "integer"}, "rating": {"type": "integer", "description": "1-5 stars"}}},
    {"name": "spotify__create_playlist", "description": "Create playlist", "parameters": {"title": {"type": "string"}}},
    {"name": "spotify__add_song_to_playlist", "description": "Add song to playlist", "parameters": {"playlist_id": {"type": "integer"}, "song_id": {"type": "integer"}}},
    {"name": "spotify__remove_song_from_playlist", "description": "Remove song from playlist", "parameters": {"playlist_id": {"type": "integer"}, "song_id": {"type": "integer"}}},
    {"name": "spotify__follow_artist", "description": "Follow an artist", "parameters": {"artist_id": {"type": "integer"}}},
    {"name": "spotify__unfollow_artist", "description": "Unfollow an artist", "parameters": {"artist_id": {"type": "integer"}}},
    {"name": "spotify__download_song", "description": "Download a song", "parameters": {"song_id": {"type": "integer"}}},
    {"name": "spotify__remove_song_from_library", "description": "Remove a song from your library", "parameters": {"song_id": {"type": "integer"}}},
    {"name": "spotify__remove_album_from_library", "description": "Remove an album from your library", "parameters": {"album_id": {"type": "integer"}}},

    # Venmo
    {"name": "venmo__login", "description": "Log in to Venmo", "parameters": {"username": {"type": "string"}, "password": {"type": "string"}}},
    {"name": "venmo__create_transaction", "description": "Send payment", "parameters": {
        "receiver_email": {"type": "string", "description": "Recipient email"},
        "amount": {"type": "number"},
        "description": {"type": "string", "description": "Payment note"},
    }},
    {"name": "venmo__show_transactions", "description": "Show transactions with filters", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "user_email": {"type": "string", "description": "Filter by other user's email"},
        "direction": {"type": "string", "description": "Filter: 'sent' or 'received'"},
        "min_created_at": {"type": "string", "description": "Min date YYYY-MM-DD"},
        "max_created_at": {"type": "string", "description": "Max date YYYY-MM-DD"},
        "min_like_count": {"type": "integer", "description": "Min like count"},
        "min_amount": {"type": "number", "description": "Min transaction amount"},
        "max_amount": {"type": "number", "description": "Max transaction amount"},
        "private": {"type": "boolean", "description": "Filter by privacy"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "venmo__show_received_payment_requests", "description": "Show payment requests received", "parameters": {
        "status": {"type": "string", "description": "Filter: 'pending', 'approved', or 'denied'"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "venmo__approve_payment_request", "description": "Approve a payment request", "parameters": {"payment_request_id": {"type": "integer"}}},
    {"name": "venmo__like_transaction", "description": "Like a transaction", "parameters": {"transaction_id": {"type": "integer"}}},
    {"name": "venmo__create_transaction_comment", "description": "Comment on a transaction", "parameters": {"transaction_id": {"type": "integer"}, "comment": {"type": "string"}}},
    {"name": "venmo__show_social_feed", "description": "Show social feed", "parameters": {
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "venmo__remind_payment_request", "description": "Send reminder for payment request", "parameters": {"payment_request_id": {"type": "integer"}}},
    {"name": "venmo__deny_payment_request", "description": "Deny a payment request", "parameters": {"payment_request_id": {"type": "integer"}}},
    {"name": "venmo__create_payment_request", "description": "Request payment from someone", "parameters": {
        "user_email": {"type": "string", "description": "Email of person to request from"},
        "amount": {"type": "number"},
        "description": {"type": "string"},
    }},
    {"name": "venmo__show_sent_payment_requests", "description": "Show payment requests you sent", "parameters": {
        "status": {"type": "string", "description": "Filter: 'pending', 'approved', or 'denied'"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "venmo__search_friends", "description": "Search your Venmo friends list", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},

    # Phone
    {"name": "phone__login", "description": "Log in to phone", "parameters": {"username": {"type": "string"}, "password": {"type": "string"}}},
    {"name": "phone__search_text_messages", "description": "Search text messages", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "phone_number": {"type": "string", "description": "Filter by contact phone number"},
        "only_latest_per_contact": {"type": "boolean", "description": "Only latest message per contact"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "phone__send_text_message", "description": "Send text message", "parameters": {"phone_number": {"type": "string"}, "message": {"type": "string"}}},
    {"name": "phone__delete_text_message", "description": "Delete text message", "parameters": {"text_message_id": {"type": "integer"}}},
    {"name": "phone__search_voice_messages", "description": "Search voice messages", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "phone_number": {"type": "string", "description": "Filter by contact phone number"},
        "only_latest_per_contact": {"type": "boolean", "description": "Only latest message per contact"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "phone__delete_voice_message", "description": "Delete voice message", "parameters": {"voice_message_id": {"type": "integer"}}},
    {"name": "phone__search_contacts", "description": "Search contacts", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "phone__show_contact_relationships", "description": "Show contact relationships (friend, coworker, roommate, etc.)", "parameters": {}},

    # File System
    {"name": "file_system__login", "description": "Log in to file system", "parameters": {"username": {"type": "string"}, "password": {"type": "string"}}},
    {"name": "file_system__show_directory", "description": "Show directory contents", "parameters": {
        "directory_path": {"type": "string", "description": "Path (default '/')"},
        "substring": {"type": "string", "description": "Filter by name substring"},
        "entry_type": {"type": "string", "description": "'all', 'files', or 'directories'"},
        "recursive": {"type": "boolean", "description": "Show recursively (default true)"},
    }},
    {"name": "file_system__show_file", "description": "Read file contents", "parameters": {"file_path": {"type": "string"}}},
    {"name": "file_system__create_file", "description": "Create a file", "parameters": {
        "file_path": {"type": "string"},
        "content": {"type": "string"},
        "overwrite": {"type": "boolean", "description": "Overwrite if exists (default false)"},
    }},
    {"name": "file_system__move_file", "description": "Move/rename a file", "parameters": {
        "source_file_path": {"type": "string"},
        "destination_file_path": {"type": "string"},
        "overwrite": {"type": "boolean", "description": "Overwrite if exists (default false)"},
    }},
    {"name": "file_system__create_directory", "description": "Create a directory", "parameters": {
        "directory_path": {"type": "string"},
        "recursive": {"type": "boolean", "description": "Create parent dirs (default false)"},
    }},
    {"name": "file_system__delete_file", "description": "Delete a file", "parameters": {"file_path": {"type": "string"}}},
    {"name": "file_system__compress_directory", "description": "Compress a directory into a zip file", "parameters": {
        "directory_path": {"type": "string"},
        "file_path": {"type": "string", "description": "Output zip file path"},
    }},

    # SimpleNote
    {"name": "simple_note__login", "description": "Log in to SimpleNote", "parameters": {"username": {"type": "string"}, "password": {"type": "string"}}},
    {"name": "simple_note__search_notes", "description": "Search notes", "parameters": {
        "query": {"type": "string", "description": "Search query"},
        "tags": {"type": "array", "description": "Filter by tags"},
        "pinned": {"type": "boolean", "description": "Filter by pinned status"},
        "page_index": {"type": "integer", "description": "Page index (default 0)"},
        "page_limit": {"type": "integer", "description": "Results per page, 1-20 (default 5)"},
    }},
    {"name": "simple_note__show_note", "description": "Show note details including full content", "parameters": {"note_id": {"type": "integer"}}},
    {"name": "simple_note__update_note", "description": "Update a note", "parameters": {
        "note_id": {"type": "integer"},
        "title": {"type": "string"},
        "content": {"type": "string"},
        "tags": {"type": "array"},
        "pinned": {"type": "boolean"},
    }},
    {"name": "simple_note__create_note", "description": "Create a note", "parameters": {
        "title": {"type": "string"},
        "content": {"type": "string"},
        "tags": {"type": "array"},
        "pinned": {"type": "boolean"},
    }},
    {"name": "simple_note__delete_note", "description": "Delete a note", "parameters": {"note_id": {"type": "integer"}}},
]


# ── WM system prompt tool schema block ──
WM_TOOL_SCHEMAS = """## TOOL SCHEMAS
All app interactions use direct tool calls in the format: app__api_name(params)
Results are paginated by default (5 per page). Use page_index and page_limit to get more.

### Supervisor APIs
- supervisor__show_account_passwords() → [{"app": "<app>", "username": "<email>", "password": "<pwd>"}, ...]
- supervisor__complete_task(answer: string) → {"status": "success"}

### Spotify APIs
- spotify__login(username, password) → {"access_token": "...", "token_type": "Bearer"}
- spotify__show_playlist_library(page_index=0, page_limit=5) → [{"playlist_id", "title", "is_public", "like_count", "song_ids": [...]}]
- spotify__show_playlist(playlist_id) → {"playlist_id", "title", "songs": [...], "rating", ...}
- spotify__search_songs(query="", artist_id=None, album_id=None, genre=None, min_release_date, max_release_date, min_play_count=0, max_play_count, min_like_count=0, page_index=0, page_limit=5) → [{"song_id", "title", "album_id", "artists": [...], "play_count", "like_count", "duration", "release_date"}]
- spotify__show_liked_songs(page_index=0, page_limit=5) → [{"song_id", "title", ...}]
- spotify__show_song_library(page_index=0, page_limit=5) → [{"song_id", "title", "album_id", "artists": [...], "duration", "added_at"}]
- spotify__show_album_library(page_index=0, page_limit=5) → [{"album_id", "title", "artists": [...], "added_at"}]
- spotify__show_album(album_id) → {"album_id", "title", "songs": [...], ...}
- spotify__show_recommendations(page_index=0, page_limit=5) → [{"song_id", "title", ...}]
- spotify__show_artist(artist_id) → {"artist_id", "name", "genre", "follower_count"}
- spotify__show_following_artists(page_index=0, page_limit=5) → [{"artist_id", "name", "genre"}]
- spotify__show_current_song() → {"song_id", "title", "artist", ...}
- spotify__next_song() → {"song_id", "title", ...}
- spotify__previous_song() → {"song_id", "title", ...}
- spotify__like_song(song_id) → {"message": "Song liked"}
- spotify__unlike_song(song_id) → {"message": "Song removed from library"}
- spotify__review_song(song_id, rating: 1-5) → {"message": "Song rated N stars"}
- spotify__create_playlist(title) → {"playlist_id", "title"}
- spotify__add_song_to_playlist(playlist_id, song_id) → {"message": "Song added"}
- spotify__remove_song_from_playlist(playlist_id, song_id) → {"message": "Song removed"}
- spotify__follow_artist(artist_id) → {"message": "Artist followed"}
- spotify__unfollow_artist(artist_id) → {"message": "Artist unfollowed"}
- spotify__download_song(song_id) → {"message": "Song downloaded"}
- spotify__remove_song_from_library(song_id) → {"message": "Song removed"}
- spotify__remove_album_from_library(album_id) → {"message": "Album removed"}

### Venmo APIs
- venmo__login(username, password) → {"access_token": "...", "token_type": "Bearer"}
- venmo__show_transactions(query="", user_email=None, direction=None["sent"|"received"], min_created_at, max_created_at, min_like_count=0, min_amount, max_amount, private=None, page_index=0, page_limit=5) → [{"transaction_id", "amount", "description", "created_at", "like_count", "sender": {...}, "receiver": {...}}]
- venmo__show_received_payment_requests(status=None["pending"|"approved"|"denied"], page_index=0, page_limit=5) → [{"payment_request_id", "sender", "amount", "description", "status"}]
- venmo__create_transaction(receiver_email, amount, description="") → {"transaction_id", ...}
- venmo__approve_payment_request(payment_request_id) → {"message": "Request approved"}
- venmo__like_transaction(transaction_id) → {"message": "Transaction liked"}
- venmo__create_transaction_comment(transaction_id, comment) → {"comment_id", ...}
- venmo__show_social_feed(page_index=0, page_limit=5) → [{"transaction_id", "sender", "amount", "description", "like_count"}]
- venmo__remind_payment_request(payment_request_id) → {"message": "Reminder sent"}
- venmo__deny_payment_request(payment_request_id) → {"message": "Request denied"}
- venmo__create_payment_request(user_email, amount, description="") → {"payment_request_id", ...}
- venmo__show_sent_payment_requests(status=None, page_index=0, page_limit=5) → [{"payment_request_id", "receiver", "amount", "status"}]
- venmo__search_friends(query="", page_index=0, page_limit=5) → [{"user_id", "name", "email"}]

### Phone APIs
- phone__login(username, password) → {"access_token": "...", "token_type": "Bearer"}
- phone__search_text_messages(query="", phone_number=None, only_latest_per_contact=false, page_index=0, page_limit=5) → [{"text_message_id", "sender_phone_number", "receiver_phone_number", "message", "created_at"}]
- phone__send_text_message(phone_number, message) → {"text_message_id", ...}
- phone__delete_text_message(text_message_id) → {"message": "Deleted"}
- phone__search_voice_messages(query="", phone_number=None, only_latest_per_contact=false, page_index=0, page_limit=5) → [{"voice_message_id", "sender_phone_number", "duration", "created_at"}]
- phone__delete_voice_message(voice_message_id) → {"message": "Deleted"}
- phone__search_contacts(query="", page_index=0, page_limit=5) → [{"contact_id", "name", "phone_number", "email"}]
- phone__show_contact_relationships() → [{"contact_id", "name", "relationship": "friend"|"coworker"|"roommate"|...}]

### File System APIs
- file_system__login(username, password) → {"access_token": "...", "token_type": "Bearer"}
- file_system__show_directory(directory_path="/", substring=None, entry_type="all", recursive=true) → [{"name", "type", "size", "path"}]
- file_system__show_file(file_path) → {"content": "..."}
- file_system__create_file(file_path, content="", overwrite=false) → {"message": "File created"}
- file_system__move_file(source_file_path, destination_file_path, overwrite=false) → {"message": "File moved"}
- file_system__create_directory(directory_path, recursive=false) → {"message": "Directory created"}
- file_system__delete_file(file_path) → {"message": "File deleted"}
- file_system__compress_directory(directory_path, file_path) → {"message": "Directory compressed"}

### SimpleNote APIs
- simple_note__login(username, password) → {"access_token": "...", "token_type": "Bearer"}
- simple_note__search_notes(query="", tags=None, pinned=None, page_index=0, page_limit=5) → [{"note_id", "title", "content", "tags", "pinned", "created_at"}]
- simple_note__show_note(note_id) → {"note_id", "title", "content", "tags", "pinned", "created_at"}
- simple_note__update_note(note_id, title=None, content=None, tags=None, pinned=None) → {"message": "Note updated"}
- simple_note__create_note(title, content, tags=None, pinned=false) → {"note_id", ...}
- simple_note__delete_note(note_id) → {"message": "Note deleted"}

### Pagination
All list endpoints return 5 results per page by default. To get all results:
- Use page_limit=20 (maximum) and increment page_index until empty results
- Or use filter params to narrow results first

### Error Responses
All tools return {"error": "<reason>"} on failure. Common errors:
- Authentication: wrong username (must be email) or wrong password → 401
- Not found: invalid ID → 404
- Validation: missing required params → 400

### Authentication Flow
1. Call supervisor__show_account_passwords() to get credentials
2. Login to each required app with the returned username/password
3. Access tokens are managed automatically — do NOT pass access_token"""


def rename_tool_in_text(text):
    """Replace old tool names with new ones in any text."""
    for old, new in NAME_MAP.items():
        text = text.replace(old, new)
    return text


def fix_system_message_tools(system_content):
    """Replace the <|tool|>...<|/tool|> block with correct tool definitions."""
    # Find and replace tool block
    tool_match = re.search(r'<\|tool\|>(.*?)<\|/tool\|>', system_content, re.DOTALL)
    if tool_match:
        new_tools = json.dumps(TOOL_DEFS)
        system_content = system_content[:tool_match.start()] + f'<|tool|>{new_tools}<|/tool|>' + system_content[tool_match.end():]
    return system_content


def fix_wm_tool_schemas(wm_prompt):
    """Replace TOOL SCHEMAS section in WM system prompt."""
    if not wm_prompt:
        return wm_prompt

    # First rename all tool names in the entire prompt
    wm_prompt = rename_tool_in_text(wm_prompt)

    # Replace TOOL SCHEMAS section
    pattern = r'## TOOL SCHEMAS.*?(?=## DOMAIN RULES|## STEERING DIRECTIVES|PREDICTION TARGET:)'
    replacement = WM_TOOL_SCHEMAS + '\n\n'
    new_wm = re.sub(pattern, replacement, wm_prompt, flags=re.DOTALL)

    if new_wm == wm_prompt and '## DOMAIN RULES' in wm_prompt:
        new_wm = wm_prompt.replace('## DOMAIN RULES', WM_TOOL_SCHEMAS + '\n\n## DOMAIN RULES')

    return new_wm


def fix_row(row):
    """Fix a single dataset row."""
    # Fix messages
    messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
    new_messages = []
    for m in messages:
        content = m.get('content', '')
        role = m['role']

        if role == 'system':
            content = fix_system_message_tools(content)
            # Also rename any tool names in system text
            content = rename_tool_in_text(content)
        elif role == 'assistant':
            # Rename tool names in tool calls
            content = rename_tool_in_text(content)
            # Also fix param names in tool call JSON
            # This is trickier - parse the tool call and fix params
        elif role == 'user':
            content = rename_tool_in_text(content)

        new_messages.append({**m, 'content': content})

    row['messages'] = json.dumps(new_messages)

    # Fix WM system prompt
    if 'wm_system_prompt' in row:
        row['wm_system_prompt'] = fix_wm_tool_schemas(row['wm_system_prompt'])

    return row


def fix_file(input_path, output_path=None):
    """Fix all rows in a JSONL file."""
    if output_path is None:
        output_path = input_path

    with open(input_path) as f:
        rows = [json.loads(l) for l in f]

    fixed = [fix_row(row) for row in rows]

    with open(output_path, 'w') as f:
        for row in fixed:
            f.write(json.dumps(row) + '\n')

    # Verify
    with open(output_path) as f:
        check = [json.loads(l) for l in f]

    # Count renames
    old_names_remaining = 0
    for row in check:
        msgs = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
        for m in msgs:
            for old_name in NAME_MAP:
                if old_name in m.get('content', ''):
                    old_names_remaining += 1

    print(f"Fixed {input_path} -> {output_path}: {len(fixed)} rows, {old_names_remaining} old names remaining")
    return old_names_remaining


if __name__ == "__main__":
    files = [
        'data/appworld_rl_split.jsonl',
        'data/appworld_test_split.jsonl',
        'data/appworld_sft_split.jsonl',
        'data/appworld_sft_gpt_agent.jsonl',
    ]
    total_remaining = 0
    for f in files:
        try:
            remaining = fix_file(f)
            total_remaining += remaining
        except FileNotFoundError:
            print(f"  SKIP {f} (not found)")

    if total_remaining > 0:
        print(f"\nWARNING: {total_remaining} old tool names still present!")
    else:
        print("\nAll tool names successfully updated.")
