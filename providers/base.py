"""Shared provider types. Agent talks only to this surface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class ModelInfo:
    id: str
    raw: Any = None
    context_window: int | None = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    prompt_tokens_details: Any = None
    cache_read_input_tokens: int = 0


@dataclass
class ToolCallFunction:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str | None
    function: ToolCallFunction
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class AssistantMessage:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    already_printed: bool = False


@dataclass
class Choice:
    message: AssistantMessage


@dataclass
class Completion:
    choices: list[Choice]
    usage: Any = None


class StreamHandler(Protocol):
    """Optional UI hooks while a provider streams a completion."""

    def thinking_start(self) -> None: ...
    def thinking_chunk(self, text: str) -> None: ...
    def thinking_end(self) -> None: ...
    def content_start(self) -> None: ...
    def content_chunk(self, text: str) -> None: ...
    def content_end(self) -> None: ...
    def tool_args_progress(self, names: str, kb: float) -> None: ...
    def stop_loading(self) -> None: ...


class LLMProvider(ABC):
    """One backend (OpenAI-compat, Gemini, …). Agent never talks to SDKs directly."""

    name: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        stream_handler: StreamHandler | None = None,
        count_usage: Callable[..., Any] | None = None,
    ) -> Completion:
        """Run one chat completion. Raises on transport/API errors."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """List models this backend exposes."""

    def is_rate_limit(self, exc: BaseException) -> bool:
        name = type(exc).__name__
        if "RateLimit" in name:
            return True
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            return True
        msg = str(exc).lower()
        return "rate limit" in msg or "429" in msg
