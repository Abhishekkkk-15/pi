"""Local Hugging Face token counting helpers (shared by commands, compaction, etc.)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tokenizers import Tokenizer

from models import Message, Role

# Hugging Face tokenizer ids used for local counting (approx. for the active provider).
PROVIDER_TOKENIZERS: Dict[str, str] = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "groq": "Xenova/gpt-4o",
}
FALLBACK_TOKENIZER = "gpt2"

_cached_tokenizer: Optional[Tokenizer] = None
_cached_tokenizer_id: Optional[str] = None


def resolve_tokenizer_id(provider: str | None = None) -> str:
    """Pick a tokenizer id from TOKENIZER_ID env, provider map, or fallback."""
    override = (os.getenv("TOKENIZER_ID") or "").strip()
    if override:
        return override
    key = (provider or "").strip().lower()
    return PROVIDER_TOKENIZERS.get(key, FALLBACK_TOKENIZER)


def load_tokenizer(provider: str | None = None) -> Tuple[Tokenizer, str]:
    """
    Load (and cache) a Hugging Face tokenizer for local counting.

    Downloads only tokenizer files from the Hub — not model weights — and
    runs encoding offline. No API cost.
    """
    global _cached_tokenizer, _cached_tokenizer_id

    tokenizer_id = resolve_tokenizer_id(provider)
    if _cached_tokenizer is not None and _cached_tokenizer_id == tokenizer_id:
        return _cached_tokenizer, tokenizer_id

    requested = tokenizer_id
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_id)
    except Exception as primary_err:
        if tokenizer_id == FALLBACK_TOKENIZER:
            raise RuntimeError(
                f"Failed to load tokenizer '{tokenizer_id}': {primary_err}"
            ) from primary_err
        try:
            tokenizer = Tokenizer.from_pretrained(FALLBACK_TOKENIZER)
            tokenizer_id = FALLBACK_TOKENIZER
        except Exception as fallback_err:
            raise RuntimeError(
                f"Failed to load tokenizer '{requested}' ({primary_err}); "
                f"fallback '{FALLBACK_TOKENIZER}' also failed ({fallback_err})"
            ) from fallback_err

    _cached_tokenizer = tokenizer
    _cached_tokenizer_id = tokenizer_id
    return tokenizer, tokenizer_id


def clear_tokenizer_cache() -> None:
    """Drop the cached tokenizer (useful in tests or after changing TOKENIZER_ID)."""
    global _cached_tokenizer, _cached_tokenizer_id
    _cached_tokenizer = None
    _cached_tokenizer_id = None


def message_to_text(message: Message) -> str:
    """Flatten a Message into text suitable for tokenization."""
    role = (
        message.role.value if isinstance(message.role, Role) else str(message.role)
    )
    parts: List[str] = [f"{role}: {message.content or ''}"]

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        try:
            parts.append(json.dumps(tool_calls, default=str, ensure_ascii=False))
        except TypeError:
            parts.append(str(tool_calls))

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        parts.append(f"tool_call_id={tool_call_id}")

    name = getattr(message, "name", None)
    if name:
        parts.append(f"name={name}")

    return "\n".join(parts)


def count_texts(
    texts: Sequence[str],
    tokenizer: Tokenizer | None = None,
    *,
    provider: str | None = None,
) -> List[int]:
    """Return per-string token counts."""
    if not texts:
        return []
    tok = tokenizer if tokenizer is not None else load_tokenizer(provider)[0]
    return [len(enc.ids) for enc in tok.encode_batch(list(texts))]


def count_messages(
    messages: Sequence[Message],
    tokenizer: Tokenizer | None = None,
    *,
    provider: str | None = None,
) -> Tuple[List[int], int, str]:
    """
    Count tokens for each message.

    Returns (per_message_counts, total, tokenizer_id).
    """
    if tokenizer is None:
        tok, tokenizer_id = load_tokenizer(provider)
    else:
        tok = tokenizer
        tokenizer_id = _cached_tokenizer_id or resolve_tokenizer_id(provider)

    counts = count_texts([message_to_text(m) for m in messages], tok)
    return counts, sum(counts), tokenizer_id


def tokens_by_role(
    messages: Sequence[Message],
    counts: Sequence[int],
) -> Dict[str, Dict[str, int]]:
    """Aggregate token and message counts keyed by role name."""
    by_role: Dict[str, Dict[str, int]] = {}
    for message, count in zip(messages, counts):
        role = (
            message.role.value if isinstance(message.role, Role) else str(message.role)
        )
        bucket = by_role.setdefault(role, {"tokens": 0, "messages": 0})
        bucket["tokens"] += count
        bucket["messages"] += 1
    return by_role


def count_history_jsonl(
    path: Path | str,
    *,
    provider: str | None = None,
    read_messages=None,
) -> Tuple[List[Message], List[int], int, str]:
    """
    Load a conversation_history.jsonl and count tokens.

    `read_messages` defaults to Memory.read_from_jsonl (imported lazily).
    """
    history_path = Path(path)
    if read_messages is None:
        from memory import Memory

        read_messages = Memory().read_from_jsonl

    messages: List[Message] = (
        read_messages(history_path) if history_path.is_file() else []
    )
    counts, total, tokenizer_id = count_messages(messages, provider=provider)
    return messages, counts, total, tokenizer_id