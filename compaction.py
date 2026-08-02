"""Conversation compaction: summarize old turns, keep recent ones intact."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

from models import Message, Role, Session
from tokenizer import count_messages


SUMMARY_PREFIX = "[Prior conversation summary]"


def _role_str(role: Any) -> str:
    if isinstance(role, Role):
        return role.value
    if isinstance(role, str):
        return role.split(".")[-1].lower()
    return str(role).lower()


def _format_message_dict(msg: dict) -> Optional[str]:
    role = str(msg.get("role", "")).upper()
    content = msg.get("content", "") or ""

    if role == "SYSTEM":
        return "[SYSTEM PROMPT INCLUDED]"

    if role == "USER":
        return f"USER:\n{content.strip()}\n"

    if role == "ASSISTANT":
        assistant_block: list[str] = []
        if content.strip():
            assistant_block.append(f"ASSISTANT:\n{content.strip()}")
        tool_calls = msg.get("tool_calls") or []
        for tool in tool_calls:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            fn_name = fn.get("name", "unknown_tool")
            fn_args = fn.get("arguments", "{}")
            assistant_block.append(f"[CALL TOOL: {fn_name}({fn_args})]")
        if assistant_block:
            return "\n".join(assistant_block) + "\n"
        return None

    if role in ("TOOL", "FUNCTION"):
        max_len = 1000
        if len(content) > max_len:
            content = (
                content[:max_len]
                + f"\n... [Truncated {len(content) - max_len} characters]"
            )
        return f"TOOL RESULT:\n{content.strip()}\n"

    return None


def messages_to_clean_transcript(messages: Sequence[Message | dict]) -> str:
    """Human-readable transcript for summarization (lossy on purpose)."""
    formatted: list[str] = []
    for item in messages:
        if isinstance(item, Message):
            msg = item.to_dict()
        else:
            msg = item
        line = _format_message_dict(msg)
        if line:
            formatted.append(line)
    return "\n---\n".join(formatted)


def jsonl_to_clean_transcript(jsonl_data: str) -> str:
    """
    Converts a JSONL string into a clean, human-readable transcript.
    Strips out system prompt bloat and formats tool calls/results clearly.
    """
    parsed: list[dict] = []
    for line in jsonl_data.strip().splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return messages_to_clean_transcript(parsed)


def build_summarizer_prompt(old_summary: str, segment_transcript: str) -> str:
    """Prompt for the compaction LLM. Always includes any prior summary."""
    prior = (old_summary or "").strip() or "(none — this is the first compaction)"
    return (
        "You are compacting a coding-agent conversation into one dense summary "
        "that will replace older turns in the model context.\n\n"
        "Include: user goals, key decisions, files created/edited, commands run, "
        "errors and fixes, open todos / next steps.\n"
        "Omit: fluff, repeated tool dumps, full file contents, system-prompt noise.\n"
        "Write plain text. Be concise but complete enough to continue the work.\n\n"
        "=== PREVIOUS SUMMARY (fold this in; do not discard) ===\n"
        f"{prior}\n\n"
        "=== NEW SEGMENT TO FOLD IN ===\n"
        f"{segment_transcript.strip()}\n\n"
        "=== UPDATED SUMMARY ===\n"
        "Write a single updated summary that replaces the previous one:\n"
    )


def find_keep_start(messages: list[Message], keep_messages: int) -> int:
    """
    Index into ``messages`` where the raw tail should begin.
    Preserves system at index 0. Aligns to a USER turn so tool chains stay intact.
    """
    n = len(messages)
    if n <= 1:
        return n

    has_system = _role_str(messages[0].role) == "system"
    body_start = 1 if has_system else 0
    body_len = n - body_start
    if body_len <= keep_messages:
        return body_start

    start = n - keep_messages
    if start < body_start:
        start = body_start

    orig = start
    while start < n and _role_str(messages[start].role) != "user":
        start += 1
    if start >= n:
        start = orig
    return max(start, body_start)


def message_to_rough_text(message: Message) -> str:
    parts = [message.content or ""]
    if getattr(message, "tool_calls", None):
        try:
            parts.append(json.dumps(message.tool_calls, default=str))
        except TypeError:
            parts.append(str(message.tool_calls))
    return "\n".join(parts)


class Compaction:
    """Decide when to compact and how to split / prompt the summarizer."""

    def __init__(
        self,
        *,
        compact_at_tokens: int = 20_000,
        keep_messages: int = 16,
        provider: str | None = None,
    ) -> None:
        self.compact_at_tokens = compact_at_tokens
        self.keep_messages = keep_messages
        self.provider = provider

    def working_messages(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> list[Message]:
        """Messages that would be sent: system + optional summary stubs + raw tail."""
        if not messages:
            return []

        summary = (session.compaction_summary if session else "") or ""
        until = int(session.compacted_until if session else 0)
        until = max(0, min(until, len(messages)))

        has_system = _role_str(messages[0].role) == "system"
        out: list[Message] = []
        if has_system:
            out.append(messages[0])

        min_body = 1 if has_system else 0
        if summary.strip() and until > min_body:
            out.append(
                Message(
                    role=Role.USER,
                    content=f"{SUMMARY_PREFIX}\n{summary.strip()}",
                )
            )
            out.append(
                Message(
                    role=Role.ASSISTANT,
                    content="Understood. I will treat that summary as prior context.",
                )
            )

        if until <= 0:
            tail_start = min_body
        else:
            tail_start = until
        out.extend(messages[tail_start:])
        return out

    def should_compact(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> bool:
        keep_start = find_keep_start(messages, self.keep_messages)
        until = int(session.compacted_until if session else 0)
        has_system = bool(messages) and _role_str(messages[0].role) == "system"
        min_until = 1 if has_system else 0
        if keep_start <= max(until, min_until):
            return False

        working = self.working_messages(messages, session)
        try:
            _, total, _ = count_messages(working, provider=self.provider)
        except Exception:
            total = sum(len(message_to_rough_text(m)) for m in working) // 4
        return total >= self.compact_at_tokens

    def plan_segment(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> Optional[tuple[list[Message], int]]:
        """
        Returns (segment_to_summarize, new_compacted_until) or None if nothing to do.
        Segment is only the *new* aged messages (not already covered by summary).
        """
        keep_start = find_keep_start(messages, self.keep_messages)
        until = int(session.compacted_until if session else 0)
        until = max(0, min(until, len(messages)))
        has_system = bool(messages) and _role_str(messages[0].role) == "system"
        min_until = 1 if has_system else 0
        if until < min_until:
            until = min_until

        if keep_start <= until:
            return None

        segment = messages[until:keep_start]
        if not segment:
            return None
        return segment, keep_start

    def build_prompt(self, old_summary: str, segment: list[Message]) -> str:
        transcript = messages_to_clean_transcript(segment)
        return build_summarizer_prompt(old_summary, transcript)


SummarizeFn = Callable[[str], str]
