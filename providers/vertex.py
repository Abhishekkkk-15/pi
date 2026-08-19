"""Google Vertex AI provider (Gemini via google-genai, vertexai=True).

Auth (any one of these):
- /login value: GCP project id, path to a service-account JSON, the JSON
  itself, or the literal `adc`
- GOOGLE_CLOUD_PROJECT + Application Default Credentials
- GOOGLE_APPLICATION_CREDENTIALS

`base_url` is the Vertex *location* (e.g. us-central1), not an OpenAI URL.
A full `https://LOCATION-aiplatform.googleapis.com` host is also accepted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from providers.base import (
    AssistantMessage,
    Choice,
    Completion,
    LLMProvider,
    ModelInfo,
    StreamHandler,
    ToolCall,
    ToolCallFunction,
)
from providers.stream import StreamUI

VERTEX_PROVIDER_NAMES = frozenset({"vertex", "vertexai"})
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
)
_THINKING_BUDGET = {"low": 1024, "medium": 8192, "high": -1}
_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
)
_SKIP_MODEL_SUBSTR = ("embed", "imagen", "imagegeneration", "veo", "lyria")


def is_vertex_provider(name: str | None) -> bool:
    return (name or "").strip().lower() in VERTEX_PROVIDER_NAMES


def parse_vertex_location(base_url: str | None) -> str:
    raw = (base_url or "").strip().rstrip("/")
    env = (os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GCLOUD_LOCATION") or "").strip()
    if not raw:
        return env or DEFAULT_LOCATION
    if "://" in raw or "aiplatform.googleapis.com" in raw:
        host = (urlparse(raw if "://" in raw else f"https://{raw}").hostname or "").lower()
        marker = "-aiplatform.googleapis.com"
        if host.endswith(marker):
            loc = host[: -len(marker)]
            return loc or env or DEFAULT_LOCATION
        if host in ("aiplatform.googleapis.com", "googleapis.com"):
            return env or DEFAULT_LOCATION
        return env or DEFAULT_LOCATION
    return raw


def _env_project() -> str:
    for key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        val = (os.getenv(key) or "").strip()
        if val:
            return val
    return ""


def _looks_like_json(secret: str) -> bool:
    s = secret.strip()
    return s.startswith("{") and s.endswith("}")


def _load_service_account(secret: str):
    from google.oauth2 import service_account

    path = Path(secret)
    if path.is_file():
        return service_account.Credentials.from_service_account_file(
            str(path), scopes=list(_SCOPES)
        )
    if _looks_like_json(secret):
        info = json.loads(secret)
        if not isinstance(info, dict):
            raise ValueError("Service account JSON must be an object")
        return service_account.Credentials.from_service_account_info(
            info, scopes=list(_SCOPES)
        )
    return None


def resolve_vertex_auth(secret: str | None) -> tuple[str, Any, str | None]:
    """Return (project, credentials_or_None, api_key_or_None)."""
    text = (secret or "").strip()
    creds = None
    api_key = None
    project = _env_project()

    if text.lower() in ("adc", "application-default"):
        text = ""

    if text.startswith("AIza"):
        api_key = text
    elif text:
        creds = _load_service_account(text)
        if creds is None:
            project = text
        else:
            pid = getattr(creds, "project_id", None)
            if pid and not project:
                project = str(pid)

    if creds is None:
        gac = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if gac:
            try:
                creds = _load_service_account(gac)
            except Exception:
                creds = None
            pid = getattr(creds, "project_id", None) if creds is not None else None
            if pid and not project:
                project = str(pid)

    if not project and creds is not None:
        project = str(getattr(creds, "project_id", "") or "")

    return project, creds, api_key


def vertex_can_configure(api_key: str | None) -> bool:
    project, creds, google_api_key = resolve_vertex_auth(api_key)
    if google_api_key:
        return True
    if project:
        return True
    if creds is not None:
        return True
    return bool(
        (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    )


def _model_id(name: str) -> str:
    return str(name).rsplit("/", 1)[-1]


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_usage(raw: Any) -> Any:
    if raw is None:
        return None
    prompt = int(
        _get_val(raw, "prompt_token_count", 0)
        or _get_val(raw, "prompt_tokens", 0)
        or 0
    )
    completion = int(
        _get_val(raw, "candidates_token_count", 0)
        or _get_val(raw, "completion_tokens", 0)
        or 0
    )
    thoughts = int(_get_val(raw, "thoughts_token_count", 0) or 0)
    completion += thoughts
    total = int(_get_val(raw, "total_token_count", 0) or (prompt + completion))
    cached = int(
        _get_val(raw, "cached_content_token_count", 0)
        or _get_val(raw, "cached_tokens", 0)
        or 0
    )

    class CompatUsage:
        prompt_tokens = prompt
        completion_tokens = completion
        total_tokens = total
        cached_tokens = cached
        prompt_tokens_details = None
        cache_read_input_tokens = cached

    return CompatUsage()


def _json_args(args: Any) -> str:
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args)
    except TypeError:
        return json.dumps({"_repr": str(args)})


class VertexAIProvider(LLMProvider):
    """Gemini on Vertex AI via `google.genai.Client(vertexai=True)`."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str | None,
        base_url: str,
    ) -> None:
        from google import genai

        self.name = name
        self.api_key = api_key or ""
        self.base_url = (base_url or "").rstrip("/")
        self.location = parse_vertex_location(self.base_url)
        project, credentials, google_api_key = resolve_vertex_auth(api_key)
        self.project = project
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "location": self.location,
        }
        if project:
            kwargs["project"] = project
        if credentials is not None:
            kwargs["credentials"] = credentials
        if google_api_key:
            kwargs["api_key"] = google_api_key
        self.client = genai.Client(**kwargs)

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
        from google.genai import types

        system_instruction, contents = self._messages_to_contents(messages, types)
        config_kwargs: dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        converted_tools = self._tools_to_vertex(tools, types)
        if converted_tools:
            config_kwargs["tools"] = converted_tools
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        thinking = self._thinking_config(types, reasoning_effort)
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking

        stream = self.client.models.generate_content_stream(
            model=_model_id(model),
            contents=contents or " ",
            config=types.GenerateContentConfig(**config_kwargs),
        )

        ui = StreamUI(stream_handler)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[str, dict[str, str]] = {}
        usage_obj: Any = None

        for chunk in stream:
            usage_obj = _get_val(chunk, "usage_metadata") or usage_obj
            candidates = _get_val(chunk, "candidates") or []
            if not candidates:
                text = _get_val(chunk, "text") or ""
                if text:
                    ui.content(text)
                    content_parts.append(text)
                continue
            parts = _get_val(_get_val(candidates[0], "content"), "parts") or []
            for part in parts:
                thought = bool(_get_val(part, "thought"))
                text = _get_val(part, "text") or ""
                fn = _get_val(part, "function_call")
                if fn:
                    name = _get_val(fn, "name") or ""
                    call_id = str(_get_val(fn, "id") or name or f"call_{len(tool_calls)}")
                    args = _json_args(_get_val(fn, "args"))
                    tool_calls[call_id] = {
                        "id": call_id,
                        "name": name,
                        "arguments": args,
                    }
                    ui.tool_progress(
                        ", ".join(t["name"] or "tool" for t in tool_calls.values()),
                        sum(len(t["arguments"]) for t in tool_calls.values()) / 1024.0,
                    )
                    continue
                if not text:
                    continue
                if thought:
                    ui.thinking(text)
                    reasoning_parts.append(text)
                else:
                    ui.content(text)
                    content_parts.append(text)

        ui.finish()
        usage_obj = _normalize_usage(usage_obj)
        if usage_obj is None and count_usage is not None:
            usage_obj = count_usage(messages, "".join(content_parts))

        final_tool_calls = [
            ToolCall(
                id=tc["id"],
                function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in tool_calls.values()
            if tc.get("name") or tc.get("arguments")
        ]
        return Completion(
            choices=[
                Choice(
                    message=AssistantMessage(
                        content="".join(content_parts) or None,
                        reasoning_content="".join(reasoning_parts) or None,
                        tool_calls=final_tool_calls or None,
                        already_printed=stream_handler is not None,
                    )
                )
            ],
            usage=usage_obj,
        )

    @staticmethod
    def _thinking_config(types: Any, effort: str | None) -> Any:
        if not effort:
            return None
        effort = effort.strip().lower()
        kwargs: dict[str, Any] = {"include_thoughts": True}
        fields = getattr(types.ThinkingConfig, "model_fields", {}) or {}
        if "thinking_level" in fields:
            kwargs["thinking_level"] = effort.upper()
        if "thinking_budget" in fields:
            kwargs["thinking_budget"] = _THINKING_BUDGET.get(effort, -1)
        try:
            return types.ThinkingConfig(**kwargs)
        except Exception:
            try:
                return types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=_THINKING_BUDGET.get(effort, -1),
                )
            except Exception:
                return types.ThinkingConfig(include_thoughts=True)

    @staticmethod
    def _tools_to_vertex(tools: list[dict[str, Any]] | None, types: Any) -> list[Any] | None:
        if not tools:
            return None
        decls: list[Any] = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict):
                continue
            schema = fn.get("parameters") or {"type": "object", "properties": {}}
            decls.append(
                types.FunctionDeclaration(
                    name=fn.get("name"),
                    description=fn.get("description") or "",
                    parameters_json_schema=schema,
                )
            )
        if not decls:
            return None
        return [types.Tool(function_declarations=decls)]

    @staticmethod
    def _messages_to_contents(
        messages: list[dict[str, Any]], types: Any
    ) -> tuple[str, list[Any]]:
        instructions: list[str] = []
        contents: list[Any] = []
        call_names: dict[str, str] = {}

        def _fn_name(msg: dict[str, Any]) -> str:
            name = msg.get("name")
            if name:
                return str(name)
            cid = str(msg.get("tool_call_id") or "")
            return call_names.get(cid, cid or "unknown")

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            if role == "system":
                if content:
                    instructions.append(str(content))
                i += 1
                continue
            if role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=str(content))],
                    )
                )
                i += 1
                continue
            if role == "assistant":
                parts: list[Any] = []
                if content:
                    parts.append(types.Part.from_text(text=str(content)))
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") if isinstance(tc, dict) else {}
                    if not isinstance(fn, dict):
                        fn = {}
                    call_id = str(tc.get("id") or "") if isinstance(tc, dict) else ""
                    name = str(fn.get("name") or "unknown")
                    if call_id:
                        call_names[call_id] = name
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {"_raw": raw_args}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    part_kwargs: dict[str, Any] = {"name": name, "args": args}
                    if call_id:
                        part_kwargs["id"] = call_id
                    try:
                        parts.append(types.Part.from_function_call(**part_kwargs))
                    except TypeError:
                        part_kwargs.pop("id", None)
                        parts.append(types.Part.from_function_call(**part_kwargs))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                i += 1
                continue
            if role == "tool":
                parts = []
                while i < len(messages) and (messages[i].get("role") or "").lower() == "tool":
                    tmsg = messages[i]
                    parts.append(
                        types.Part.from_function_response(
                            name=_fn_name(tmsg),
                            response={"output": str(tmsg.get("content") or "")},
                        )
                    )
                    i += 1
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
                continue
            i += 1
        return "\n\n".join(instructions), contents

    def list_models(self) -> list[ModelInfo]:
        results: list[ModelInfo] = []
        seen: set[str] = set()
        try:
            pager = self.client.models.list(config={"page_size": 100})
            for m in pager:
                name = _get_val(m, "name") or _get_val(m, "id") or ""
                mid = _model_id(str(name))
                low = mid.lower()
                if not mid or mid in seen:
                    continue
                if any(s in low for s in _SKIP_MODEL_SUBSTR):
                    continue
                seen.add(mid)
                results.append(ModelInfo(id=mid, raw=m))
        except Exception:
            results = []
        if not results:
            for mid in DEFAULT_MODELS:
                results.append(ModelInfo(id=mid))
        return results

    def is_rate_limit(self, exc: BaseException) -> bool:
        if super().is_rate_limit(exc):
            return True
        name = type(exc).__name__
        if "ResourceExhausted" in name or "TooManyRequests" in name:
            return True
        msg = str(exc).lower()
        return "resource exhausted" in msg or ("quota" in msg and "exceed" in msg)
