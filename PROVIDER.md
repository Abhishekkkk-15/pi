# Provider layer

The Agent talks to language models through a small Python interface. HTTP, SDKs, streaming event names, and request-shape differences stay inside `providers/`. `llm.py` never imports a vendor SDK.

## Responsibilities

**Agent (`llm.py`, `config.py`, `commands.py`) owns:**

- Session history, compaction, permissions, tool execution
- Console streaming UI (markdown, thinking gutter, loading spinner)
- Auth and active provider/model (`auth.json`, `/login`, `/model`)
- `/reasoning` (stores `low` / `medium` / `high` or unset)
- 429 handling: rotate secondary API key and retry once

**Provider owns:**

- Constructing the vendor client
- Mapping Agent messages + Chat Completions-style tools to the vendor API
- Streaming tokens and reconstructing one `Completion`
- Listing models
- Deciding whether an exception is a rate limit (optional override)

## Files

| Path | Purpose |
|------|---------|
| `providers/base.py` | `LLMProvider` ABC and shared types |
| `providers/openai.py` | Official OpenAI (Responses API) + OpenAI-compatible Chat Completions |
| `providers/__init__.py` | `create_provider(name, api_key, base_url)` factory |
| `config.py` | `BUILTIN_PROVIDERS` names, default URLs, default models |
| `llm.py` | `Agent.create_model()` → `self.llm.complete(...)` |
| `pyproject.toml` | `packages = ["prompts", "providers"]` |

Custom backends are extra modules under `providers/` (for example `providers/gemini.py`). Register them in the factory. Do not put SDK calls in `llm.py`.

## How a request flows

```
/login + auth.json          config.provider, api_key, base_url, model
        │
        ▼
Agent.create_model()        create_provider(name, api_key, base_url)
        │
        ▼
self.llm                    LLMProvider instance (or None if no key)
        │
Agent._create_completion()
        │
        ├─ tools            Chat Completions function schema (or None)
        ├─ messages         OpenAI-style dicts (system/user/assistant/tool)
        ├─ reasoning_effort config.reasoning_effort  (None or low|medium|high)
        ├─ max_tokens       config.max_tokens
        ├─ stream_handler   console hooks (None if stream_to_ui is False)
        └─ count_usage      tokenizer fallback if the API omits usage
        │
        ▼
provider.complete(...)      vendor HTTP + stream reconstruction
        │
        ▼
Completion                  choices[0].message + usage
        │
Agent                       persist history, run tools, print if not already streamed
```

`create_provider` returns `None` when `api_key` is missing. The Agent then tells the user to run `/login`.

Factory today:

```python
# providers/__init__.py
if not api_key:
    return None
return OpenAIProvider(name=name, api_key=api_key, base_url=base_url)
```

Mistral, Groq, OpenRouter, and custom OpenAI-compatible URLs all go through `OpenAIProvider` because they speak Chat Completions. A native Gemini (or Anthropic) backend needs its own class.

## Contract: `LLMProvider`

```python
class LLMProvider(ABC):
    name: str

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
        """One generation. Raise on transport/API errors."""

    def list_models(self) -> list[ModelInfo]:
        """Models this backend exposes (used by /model)."""

    def is_rate_limit(self, exc: BaseException) -> bool:
        """Default: RateLimit* type name, status 429, or 'rate limit'/'429' in str(exc)."""
```

Transport and API errors must **raise**. The Agent catches them, checks `is_rate_limit`, and either retries with another key or surfaces the error.

### `complete` arguments

| Argument | Meaning |
|----------|---------|
| `messages` | Chat history in OpenAI Chat Completions shape (see below) |
| `model` | Active model id from config (`/model`) |
| `tools` | `None` or a list of `{type: "function", function: {name, description, parameters}}` |
| `max_tokens` | Cap, or `None` to omit |
| `reasoning_effort` | `None` (do not send reasoning params) or `"low"` / `"medium"` / `"high"` |
| `stream_handler` | Live UI; `None` means reconstruct silently (no console stream) |
| `count_usage` | `count_usage(messages, completion_text)` if the vendor did not return usage |

There is **no** `is_reasoning` flag and **no** model-name sniffing (`o1`, `o3`, `gpt-5`, …). If the user set `/reasoning`, the provider forwards that effort. If the model cannot use it, the vendor API errors (compat Chat Completions drops `reasoning_effort` on a retry if the host rejects extra kwargs).

### Inbound `messages`

Roles the Agent sends:

| `role` | Fields |
|--------|--------|
| `system` | `content` — full system prompt |
| `user` | `content` |
| `assistant` | `content` and/or `tool_calls` |
| `tool` | `content` (tool result), `tool_call_id` matching the assistant tool call `id` |

Assistant `tool_calls` entries are dicts with `.model_dump()` shape:

```python
{
    "id": "call_...",
    "type": "function",
    "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"},
}
```

Convert this list to whatever your vendor needs. Do not change how the Agent stores history.

### Inbound `tools`

Chat Completions function tools:

```python
{
    "type": "function",
    "function": {
        "name": "read",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}, "required": [...]},
    },
}
```

Official OpenAI Responses expects a flatter tool object (`type`, `name`, `description`, `parameters`). `OpenAIProvider` converts that internally.

### Return type: `Completion`

```python
Completion(
    choices=[
        Choice(
            message=AssistantMessage(
                content=str | None,              # visible assistant text
                reasoning_content=str | None,    # thinking / reasoning stream
                tool_calls=list[ToolCall] | None,
                already_printed=bool,            # True if stream_handler was used
            )
        )
    ],
    usage=...,  # duck-typed; see Usage
)
```

`ToolCall` must implement `model_dump()` (the dataclass in `base.py` already does). The Agent does:

```python
tool_calls_dicts = [tc.model_dump() for tc in llm_res.message.tool_calls]
```

Set `already_printed=True` when you drove `stream_handler` so the Agent does not reprint thinking/content. Set it `False` when `stream_handler` was `None`.

Empty text should be `None`, not `""`. Empty tool lists should be `None`.

### `usage`

The Agent reads either dict keys or attributes:

- `prompt_tokens` (required for cost / `/tokens`)
- `completion_tokens`
- `total_tokens` (optional; otherwise prompt + completion)
- `cached_tokens` and/or `prompt_tokens_details.cached_tokens`
- `cache_read_input_tokens` / `prompt_cache_hit_tokens` / `cached_content_token_count` as fallbacks

`OpenAIProvider` maps Responses `input_tokens` / `output_tokens` onto `prompt_tokens` / `completion_tokens`. If usage is missing, call `count_usage` when it was passed.

`providers.base.Usage` is the preferred dataclass; duck-typed objects are fine.

### `StreamHandler`

Call these in order while streaming. Skip a family if that stream never starts.

| Method | When |
|--------|------|
| `thinking_start()` | First reasoning/thinking delta |
| `thinking_chunk(text)` | Each thinking delta |
| `thinking_end()` | Thinking finished (before content or tools) |
| `content_start()` | First visible text delta |
| `content_chunk(text)` | Each text delta |
| `content_end()` | Text finished |
| `tool_args_progress(names, kb)` | While function-call arguments accumulate (`names` is a comma-separated label, `kb` is argument bytes / 1024) |
| `stop_loading()` | Leave the tool-args spinner (before content, or at end) |

If `stream_handler` is `None`, still assemble the full `Completion`; just do not call UI hooks.

`OpenAIProvider` uses `_StreamUI` to keep start/end pairing correct (thinking closed before content, content closed before tool progress).

### `list_models`

Return `list[ModelInfo]` with at least `id`. `raw` can hold the vendor payload. `context_window` is optional.

`OpenAIProvider` GETs `{base_url}/models` with `Authorization: Bearer …` (4s timeout), then falls back to `client.models.list()`.

### `is_rate_limit`

Override if your SDK uses a different exception type. Default covers OpenAI-style `RateLimitError`, `status_code == 429`, and the strings `rate limit` / `429`.

On True, the Agent rotates to the secondary key (if configured) via `rotate_provider_key`, rebuilds the provider with `apply_provider_runtime()`, and retries **once**.

## OpenAI provider (`OpenAIProvider`)

One class, two HTTP APIs.

### Which API

`uses_responses_api()` is **host-based**, not model-based:

- `True` if `name == "openai"` (case-insensitive)
- `True` if hostname is `api.openai.com` or `*.api.openai.com`
- `False` for Groq, Mistral, OpenRouter, Azure-compat, local proxies, custom `/login` URLs

Compat hosts generally do **not** implement `POST /v1/responses`.

### Official OpenAI: Responses API

Call:

```python
self.client.responses.create(...)
```

Not `client.chat.completions.create`, and not `client.chat.responses.create` (that method does not exist).

Reasoning on current OpenAI models is configured with:

```python
kwargs["reasoning"] = {"effort": reasoning_effort}  # only if reasoning_effort is set
```

That parameter belongs on Responses, not Chat Completions.

Other mappings:

| Agent | Responses |
|-------|-----------|
| system messages | `instructions` (joined with `\n\n`) |
| user / assistant / tool | `input` list |
| assistant `tool_calls` | `{type: "function_call", call_id, name, arguments}` |
| `role: tool` | `{type: "function_call_output", call_id, output}` |
| Chat Completions tools | `{type: "function", name, description, parameters}` |
| `max_tokens` | `max_output_tokens` |
| streaming | `stream=True` |
| persistence | `store=False` |

Stream events handled:

| Event `type` | Effect |
|--------------|--------|
| `response.output_text.delta` / `response.text.delta` | Assistant content |
| `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` | Thinking UI + `reasoning_content` |
| `response.output_item.added` (`function_call`) | Start a tool call |
| `response.function_call_arguments.delta` / `.done` | Accumulate tool JSON |
| `response.completed` | Usage from `response.usage` |

Tool `id` stored for the Agent is Responses `call_id` (needed so later `function_call_output` items match).

### Compat hosts: Chat Completions

Call:

```python
self.client.chat.completions.create(
    model=...,
    messages=messages,   # passed through
    tools=tools,         # passed through
    stream=True,
    stream_options={"include_usage": True},
    max_tokens=...,      # if set
)
```

If `reasoning_effort` is set:

- send `reasoning_effort=...`
- if `name == "openrouter"`, also send `include_reasoning=True`

If that request fails, retry **without** `stream_options` and `reasoning_effort` so strict-compat servers still work.

Stream deltas:

- `delta.content` → content
- `delta.reasoning_content` or `delta.reasoning` → thinking
- `delta.tool_calls` (indexed) → reconstructed `ToolCall` list

### Reasoning summary

| Setting | OpenAI Responses | Compat Chat Completions |
|---------|------------------|-------------------------|
| `/reasoning` unset (`None`) | omit `reasoning` | omit `reasoning_effort` |
| `low` / `medium` / `high` | `reasoning={"effort": ...}` | `reasoning_effort=...` |

No `looks_like_reasoning_model`. Effort is forwarded whenever the user enabled it.

## Config and builtins

`config.BUILTIN_PROVIDERS`:

| Name | Default `base_url` | Default model | HTTP via |
|------|--------------------|---------------|----------|
| `openai` | `https://api.openai.com/v1` | `gpt-4o` | Responses |
| `mistral` | `https://api.mistral.ai/v1` | `mistral-large-latest` | Chat Completions |
| `groq` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Chat Completions |

Anything else (OpenRouter, local llama.cpp, custom) is a custom provider **name** in `auth.json` with its own `base_url` and keys. Until you register a dedicated class, the factory still builds `OpenAIProvider`, so the URL must be OpenAI-compatible.

Keys, active provider, and model live under `llm_settings` in `auth.json` (see `config.py`). `/reasoning` is stored in `app_settings.reasoning_effort` (`low` / `medium` / `high` / unset). Env fallback: `REASONING_EFFORT`.

## Adding a new native provider

Example: Gemini.

### 1. Implement `providers/gemini.py`

Subclass `LLMProvider`. Convert OpenAI-style `messages` / `tools` to Gemini, stream into `StreamHandler`, return `Completion`.

```python
from providers.base import (
    LLMProvider,
    Completion,
    Choice,
    AssistantMessage,
    ToolCall,
    ToolCallFunction,
    ModelInfo,
)

class GeminiProvider(LLMProvider):
    def __init__(self, *, name: str, api_key: str, base_url: str) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # self.client = ... vendor SDK ...

    def complete(self, messages, *, model, tools=None, max_tokens=None,
                 reasoning_effort=None, stream_handler=None, count_usage=None) -> Completion:
        ...

    def list_models(self) -> list[ModelInfo]:
        ...
```

Checklist for `complete`:

1. Map `system` / `user` / `assistant` / `tool` into the vendor history format.
2. Map Chat Completions tools into vendor function declarations.
3. If `reasoning_effort` is set, use the vendor’s thinking/reasoning parameter; if unset, omit it.
4. Stream: thinking → content → tool-arg progress; always close open UI sections.
5. Build `ToolCall(id=..., function=ToolCallFunction(name=..., arguments=...))` with ids the next turn can round-trip as `tool_call_id`.
6. Normalize usage to `prompt_tokens` / `completion_tokens`.
7. `already_printed = stream_handler is not None`.
8. Raise vendor errors; do not swallow 429 (the Agent retries).

### 2. Register in `providers/__init__.py`

```python
from providers.gemini import GeminiProvider

def create_provider(*, name: str, api_key: str | None, base_url: str) -> LLMProvider | None:
    if not api_key:
        return None
    if name.lower() == "gemini":
        return GeminiProvider(name=name, api_key=api_key, base_url=base_url)
    return OpenAIProvider(name=name, api_key=api_key, base_url=base_url)
```

Match on `name` and/or hostname if the same class should not handle every URL.

### 3. Optional builtin in `config.py`

```python
BUILTIN_PROVIDERS: dict[str, dict[str, str]] = {
    # ...
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com",
        "default_model": "gemini-2.0-flash",
    },
}
```

Without this, users can still `/login` a custom provider name; they must type URL and model themselves.

### 4. Dependency

Add the vendor SDK in `pyproject.toml` if needed. `openai` is already a dependency because `OpenAIProvider` uses it.

### 5. Do not change `llm.py` for the SDK

`create_model()` already passes `name`, `api_key`, and `base_url`. New providers plug in at the factory.

## What you should not put in a provider

- Permission prompts, tool execution, or file I/O (Agent + `tools.py`)
- Compaction, history stubbing, session JSONL
- Console widgets (only call `StreamHandler`)
- `/login` UI (config + commands)
- Catch-and-ignore of 429 (raise so key rotation can run)

## Testing a provider locally

Smoke-import without a live key:

```python
from providers.openai import OpenAIProvider

o = OpenAIProvider(name="openai", api_key="x", base_url="https://api.openai.com/v1")
assert o.uses_responses_api() is True

g = OpenAIProvider(name="groq", api_key="x", base_url="https://api.groq.com/openai/v1")
assert g.uses_responses_api() is False
```

End-to-end: `/login` that provider, `/model`, send a prompt, trigger a tool (e.g. read a file), then `/reasoning medium` on a model that supports it and confirm thinking streams in the console.
