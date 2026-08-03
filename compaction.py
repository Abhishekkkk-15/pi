"""Conversation compaction: summarize old turns, keep recent ones intact."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence, Final

from models import Message, Role, Session
from tokenizer import count_messages


SUMMARY_PREFIX = "[Prior conversation summary]"

# PI's github path for compaction : https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/compaction/compaction.ts

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
            if not fn:
                continue
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


# def build_summarizer_prompt(old_summary: str, segment_transcript: str) -> str:
#     """Prompt for the compaction LLM. Always includes any prior summary."""
#     prior = (old_summary or "")
#     if not prior:
#         print("not prior summary")
#     return (
#         "You are compacting a coding-agent conversation into one dense summary "
#         "that will replace older turns in the model context.\n\n"
#         "Include: user goals, key decisions, files created/edited, commands run, "
#         "errors and fixes, open todos / next steps.\n"
#         "Omit: fluff, repeated tool dumps, full file contents, system-prompt noise.\n"
#         "Write plain text. Be concise but complete enough to continue the work.\n\n"
#         "=== PREVIOUS SUMMARY (fold this in; do not discard) ===\n"
#         f"{prior}\n\n"
#         "=== NEW SEGMENT TO FOLD IN ===\n"
#         f"{segment_transcript.strip()}\n\n"
#         "=== UPDATED SUMMARY ===\n"
#         "Write a single updated summary that replaces the previous one:\n"
#     )

SUMMARIZE_PROMPT: Final[str]= """
The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages
"""

UPDATE_SUMMARY_PROMPT :Final[str] = """
The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.
"""

def build_summarizer_prompt(old_summary: str, segment_transcript: str) -> str:
    """Prompt for the compaction LLM. Always includes any prior summary."""
    prior = (old_summary or "").strip()
    base_prompt = UPDATE_SUMMARY_PROMPT if prior else SUMMARIZE_PROMPT

    # Conversation first so "messages above" in the template refers to real turns.
    # Use f-strings / concatenation — ${...} is JavaScript and stays literal in Python.
    parts = [
        "<conversation>\n",
        segment_transcript.strip(),
        "\n</conversation>\n\n",
    ]
    if prior:
        parts.extend(
            [
                "<previous-summary>\n",
                prior,
                "\n</previous-summary>\n\n",
            ]
        )
    parts.append(base_prompt.strip())
    parts.append("\n")
    return "".join(parts)


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


# When over token budget but message count is still inside the normal keep
# window (few huge turns), shrink keep down to this floor so we can still fold.
MIN_KEEP_ON_OVERFLOW = 2


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

    def working_token_count(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> int:
        working = self.working_messages(messages, session)
        try:
            _, total, _ = count_messages(working, provider=self.provider)
            return total
        except Exception:
            return sum(len(message_to_rough_text(m)) for m in working) // 4

    def over_token_budget(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> bool:
        return self.working_token_count(messages, session) >= self.compact_at_tokens

    def _foldable(
        self,
        messages: list[Message],
        session: Session | None,
        keep_messages: int,
    ) -> bool:
        keep_start = find_keep_start(messages, keep_messages)
        until = int(session.compacted_until if session else 0)
        has_system = bool(messages) and _role_str(messages[0].role) == "system"
        min_until = 1 if has_system else 0
        return keep_start > max(until, min_until)

    def effective_keep_messages(
        self,
        messages: list[Message],
        session: Session | None,
        *,
        allow_shrink: bool,
    ) -> int:
        """
        Keep-window size for this pass.

        If the normal keep window leaves nothing to fold but ``allow_shrink``
        is set (over token budget / manual force), shrink keep down toward
        MIN_KEEP_ON_OVERFLOW so large-but-few turns can still be summarized.
        """
        normal = max(self.keep_messages, MIN_KEEP_ON_OVERFLOW)
        if self._foldable(messages, session, normal):
            return normal
        if not allow_shrink:
            return normal

        has_system = bool(messages) and _role_str(messages[0].role) == "system"
        body_len = len(messages) - (1 if has_system else 0)
        # Need at least one message outside the keep tail
        for keep in range(min(normal, body_len) - 1, MIN_KEEP_ON_OVERFLOW - 1, -1):
            if keep < MIN_KEEP_ON_OVERFLOW:
                break
            if self._foldable(messages, session, keep):
                return keep
        return MIN_KEEP_ON_OVERFLOW

    def should_compact(
        self,
        messages: list[Message],
        session: Session | None,
    ) -> bool:
        """Auto-trigger: over token budget and a foldable segment exists (maybe with shrunk keep)."""
        if not self.over_token_budget(messages, session):
            return False
        keep = self.effective_keep_messages(messages, session, allow_shrink=True)
        return self._foldable(messages, session, keep)

    def plan_segment(
        self,
        messages: list[Message],
        session: Session | None,
        *,
        keep_messages: int | None = None,
    ) -> Optional[tuple[list[Message], int, int]]:
        """
        Returns (segment_to_summarize, new_compacted_until, keep_used) or None.
        Segment is only the *new* aged messages (not already covered by summary).
        """
        keep = self.keep_messages if keep_messages is None else keep_messages
        keep_start = find_keep_start(messages, keep)
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
        return segment, keep_start, keep

    def build_prompt(self, old_summary: str, segment: list[Message]) -> str:
        transcript = messages_to_clean_transcript(segment)
        return build_summarizer_prompt(old_summary, transcript)


SummarizeFn = Callable[[str], str]
