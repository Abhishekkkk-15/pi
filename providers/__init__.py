"""Factory for LLM backends.

To add a provider (e.g. Gemini):
1. Create `providers/gemini.py` with a class that subclasses `LLMProvider`
   and implements `complete()` + `list_models()`.
2. Register it in `create_provider()` below (match `name` / base_url).
3. Optionally add a builtin entry in `config.BUILTIN_PROVIDERS`.
4. Do not put SDK calls in `llm.py` — Agent only uses `self.llm.complete(...)`.
"""

from providers.base import Completion, LLMProvider, ModelInfo, StreamHandler
from providers.openai import OpenAIProvider


def create_provider(
    *,
    name: str,
    api_key: str | None,
    base_url: str,
) -> LLMProvider | None:
    """Build the active backend."""
    if not api_key:
        return None
    # Gemini would be: if name.lower() == "gemini": return GeminiProvider(...)
    return OpenAIProvider(name=name, api_key=api_key, base_url=base_url)


__all__ = [
    "Completion",
    "LLMProvider",
    "ModelInfo",
    "OpenAIProvider",
    "StreamHandler",
    "create_provider",
]
