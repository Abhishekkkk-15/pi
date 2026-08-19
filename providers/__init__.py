"""Factory for LLM backends.

To add a provider (e.g. Anthropic):
1. Create `providers/<name>.py` with a class that subclasses `LLMProvider`
   and implements `complete()` + `list_models()`.
2. Register it in `create_provider()` below (match `name` / base_url).
3. Optionally add a builtin entry in `config.BUILTIN_PROVIDERS`.
4. Do not put SDK calls in `llm.py` — Agent only uses `self.llm.complete(...)`.
"""

from providers.base import Completion, LLMProvider, ModelInfo, StreamHandler
from providers.openai import OpenAIProvider
from providers.vertex import (
    VertexAIProvider,
    is_vertex_provider,
    vertex_can_configure,
)


def create_provider(
    *,
    name: str,
    api_key: str | None,
    base_url: str,
) -> LLMProvider | None:
    """Build the active backend."""
    if is_vertex_provider(name):
        if not vertex_can_configure(api_key):
            return None
        return VertexAIProvider(name=name, api_key=api_key, base_url=base_url)
    if not api_key:
        return None
    return OpenAIProvider(name=name, api_key=api_key, base_url=base_url)


__all__ = [
    "Completion",
    "LLMProvider",
    "ModelInfo",
    "OpenAIProvider",
    "StreamHandler",
    "VertexAIProvider",
    "create_provider",
]
