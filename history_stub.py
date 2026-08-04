"""Shrink conversation history payloads for write/edit/read/bash dumps.

Strategy:
- After a successful write/edit, replace fat assistant tool_call arguments with
  a short summary (path + line counts). Tool *results* for write/edit were
  already short; we enrich them in tools.py.
- Keep fresh read/bash/grep/web_search results for the recent (hot) window.
- Age out older large tool results to stubs so the model re-reads/re-runs
  instead of replaying megabytes of context.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from models import Message, Role

STUB_MARKER = "[omitted"
STUBBABLE_RESULT_TOOLS = frozenset({"read", "bash", "grep", "web_search"})
WRITE_EDIT_TOOLS = frozenset({"write", "edit"})

# Age-out thresholds for tool *results* outside the hot window
AGE_OUT_MIN_CHARS = 800
AGE_OUT_PREVIEW_CHARS = 120


def line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def byte_count(text: str) -> int:
    if not text:
        return 0
    return len(text.encode("utf-8"))


def is_stubbed(text: str) -> bool:
    return bool(text) and STUB_MARKER in text[:160]


def tool_succeeded(result: str) -> bool:
    """True when the tool result is not an error / denial."""
    if not result:
        return False
    head = result.lstrip()[:48].lower()
    if head.startswith("error"):
        return False
    if "permission denied" in head:
        return False
    return True


def stub_write_arguments(args: dict[str, Any]) -> dict[str, Any]:
    path = str(args.get("path", "") or "")
    content = str(args.get("content", "") or "")
    if is_stubbed(content):
        return args
    lines = line_count(content)
    nbytes = byte_count(content)
    return {
        "path": path,
        "content": (
            f"{STUB_MARKER}: wrote {lines} lines, {nbytes} bytes - "
            "re-read the file if you need the contents]"
        ),
    }


def stub_edit_arguments(args: dict[str, Any]) -> dict[str, Any]:
    path = str(args.get("path", "") or "")
    edits = args.get("edits") if isinstance(args.get("edits"), list) else []
    if not edits:
        return {"path": path, "edits": []}

    # Already stubbed?
    first = edits[0] if isinstance(edits[0], dict) else {}
    if is_stubbed(str(first.get("oldText", "") or "")):
        return args

    old_lines = 0
    new_lines = 0
    stubbed_hunks: list[dict[str, str]] = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            continue
        old_t = str(edit.get("oldText", "") or "")
        new_t = str(edit.get("newText", "") or "")
        ol = line_count(old_t)
        nl = line_count(new_t)
        old_lines += ol
        new_lines += nl
        stubbed_hunks.append(
            {
                "oldText": f"{STUB_MARKER} hunk {i + 1}: {ol} lines]",
                "newText": f"{STUB_MARKER} hunk {i + 1}: {nl} lines]",
            }
        )

    return {
        "path": path,
        "edits": stubbed_hunks
        or [
            {
                "oldText": (
                    f"{STUB_MARKER}: {len(edits)} hunk(s), "
                    f"-{old_lines}/+{new_lines} lines - re-read if needed]"
                ),
                "newText": "[omitted]",
            }
        ],
    }


def _tool_call_id(tc: Any) -> Optional[str]:
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


def _tool_call_function(tc: Any) -> Optional[dict[str, Any]]:
    if isinstance(tc, dict):
        fn = tc.get("function")
        return fn if isinstance(fn, dict) else None
    fn = getattr(tc, "function", None)
    if fn is None:
        return None
    if isinstance(fn, dict):
        return fn
    return {
        "name": getattr(fn, "name", None),
        "arguments": getattr(fn, "arguments", None),
    }


def stub_assistant_tool_call(
    assistant: Message,
    *,
    tool_call_id: Optional[str],
    tool_name: str,
) -> bool:
    """
    Replace fat write/edit arguments on the in-memory assistant message.
    Returns True if anything changed.
    """
    if tool_name not in WRITE_EDIT_TOOLS:
        return False
    calls = getattr(assistant, "tool_calls", None)
    if not calls:
        return False

    changed = False
    for tc in calls:
        if tool_call_id and _tool_call_id(tc) != tool_call_id:
            continue
        fn = _tool_call_function(tc)
        if not fn:
            continue
        name = fn.get("name") or tool_name
        if name not in WRITE_EDIT_TOOLS:
            continue
        raw = fn.get("arguments", "") or ""
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(args, dict):
            continue

        if name == "write":
            stubbed = stub_write_arguments(args)
        else:
            stubbed = stub_edit_arguments(args)

        new_raw = json.dumps(stubbed, ensure_ascii=False)
        if new_raw == raw:
            continue
        fn["arguments"] = new_raw
        if isinstance(tc, dict):
            tc["function"] = fn
        changed = True

    return changed


def stub_tool_result(name: str, content: str) -> str:
    """Collapse a large aged tool result to a short re-fetch hint."""
    if not content or is_stubbed(content):
        return content
    if not tool_succeeded(content):
        return content

    lines = line_count(content)
    nbytes = byte_count(content)
    preview = content[:AGE_OUT_PREVIEW_CHARS].replace("\n", " ").strip()
    if len(content) > AGE_OUT_PREVIEW_CHARS:
        preview += "…"
    return (
        f"{STUB_MARKER} {name} result: {lines} lines, {nbytes} bytes - "
        f"re-read or re-run if needed]\n"
        f"preview: {preview}"
    )


def age_out_large_payloads(
    messages: list[Message],
    *,
    keep_recent: int = 16,
) -> bool:
    """
    Stub aged tool results (and any leftover fat write/edit args) outside the
    hot window. Hot window = last `keep_recent` non-system messages.
    Returns True if anything changed.
    """
    if not messages or keep_recent < 1:
        return False

    # Indices of non-system messages from oldest to newest
    body_indices = [
        i
        for i, m in enumerate(messages)
        if (m.role.value if isinstance(m.role, Role) else str(m.role)).lower()
        != "system"
    ]
    if not body_indices:
        return False

    hot_start_pos = max(0, len(body_indices) - keep_recent)
    hot_index_set = set(body_indices[hot_start_pos:])

    changed = False
    for i, msg in enumerate(messages):
        role = (msg.role.value if isinstance(msg.role, Role) else str(msg.role)).lower()

        # Always safe to stub write/edit args once the tool already ran
        # (even in the hot window — model should re-read the file).
        if role == "assistant" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls or []:
                fn = _tool_call_function(tc)
                if not fn:
                    continue
                name = str(fn.get("name") or "")
                if name not in WRITE_EDIT_TOOLS:
                    continue
                raw = fn.get("arguments", "") or ""
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(args, dict):
                    continue
                stubbed = (
                    stub_write_arguments(args)
                    if name == "write"
                    else stub_edit_arguments(args)
                )
                new_raw = json.dumps(stubbed, ensure_ascii=False)
                if new_raw != raw:
                    fn["arguments"] = new_raw
                    if isinstance(tc, dict):
                        tc["function"] = fn
                    changed = True

        if i in hot_index_set:
            continue

        if role != "tool":
            continue
        name = (getattr(msg, "name", None) or "tool").lower()
        if name not in STUBBABLE_RESULT_TOOLS:
            continue
        content = msg.content or ""
        if len(content) < AGE_OUT_MIN_CHARS or is_stubbed(content):
            continue
        new_content = stub_tool_result(name, content)
        if new_content != content:
            msg.content = new_content
            changed = True

    return changed
