"""OpenAI provider.

Official OpenAI (`api.openai.com`) uses the Responses API:
`client.responses.create` — required for reasoning (`reasoning.effort`).

Groq / Mistral / OpenRouter / custom stay on Chat Completions:
`client.chat.completions.create`.
"""

from __future__ import annotations

import json
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openai import OpenAI

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
from providers.stream import StreamUI as _StreamUI


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_usage(raw: Any) -> Any:
    """Map Responses `input_tokens` / Chat `prompt_tokens` into one shape for Agent."""
    if raw is None:
        return None

    prompt = int(
        _get_val(raw, "prompt_tokens", 0)
        or _get_val(raw, "input_tokens", 0)
        or 0
    )
    completion = int(
        _get_val(raw, "completion_tokens", 0)
        or _get_val(raw, "output_tokens", 0)
        or 0
    )
    total = int(_get_val(raw, "total_tokens", 0) or (prompt + completion))
    cached = int(_get_val(raw, "cached_tokens", 0) or 0)
    if not cached:
        details = (
            _get_val(raw, "prompt_tokens_details")
            or _get_val(raw, "input_tokens_details")
        )
        cached = int(_get_val(details, "cached_tokens", 0) or 0)
    if not cached:
        cached = int(_get_val(raw, "cache_read_input_tokens", 0) or 0)

    class CompatUsage:
        prompt_tokens = prompt
        completion_tokens = completion
        total_tokens = total
        cached_tokens = cached
        prompt_tokens_details = None
        cache_read_input_tokens = cached

    return CompatUsage()


class OpenAIProvider(LLMProvider):
    """Official OpenAI (Responses) + OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def uses_responses_api(self) -> bool:
        """True for api.openai.com. Compat hosts do not implement /v1/responses."""
        if (self.name or "").lower() == "openai":
            return True
        host = (urlparse(self.base_url).hostname or "").lower()
        return host == "api.openai.com" or host.endswith(".api.openai.com")

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
        if self.uses_responses_api():
            return self._complete_responses(
                messages,
                model=model,
                tools=tools,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                stream_handler=stream_handler,
                count_usage=count_usage,
            )
        return self._complete_chat(
            messages,
            model=model,
            tools=tools,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            stream_handler=stream_handler,
            count_usage=count_usage,
        )

    # ------------------------------------------------------------------
    # Responses API (official OpenAI)
    # ------------------------------------------------------------------

    def _complete_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        stream_handler: StreamHandler | None,
        count_usage: Callable[..., Any] | None,
    ) -> Completion:
        instructions, input_items = self._messages_to_responses_input(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items or [{"role": "user", "content": ""}],
            "stream": True,
            "store": False,
        }
        if instructions:
            kwargs["instructions"] = instructions
        converted_tools = self._tools_to_responses(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        stream = self.client.responses.create(**kwargs)
        ui = _StreamUI(stream_handler)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[str, dict[str, str]] = {}
        usage_obj: Any = None

        def _tool_key(event: Any) -> str:
            return str(
                _get_val(event, "item_id")
                or _get_val(event, "output_index")
                or len(tool_calls)
            )

        for event in stream:
            etype = _get_val(event, "type") or ""

            if etype in (
                "response.output_text.delta",
                "response.text.delta",
            ):
                delta = _get_val(event, "delta") or ""
                ui.content(delta)
                content_parts.append(delta)

            elif etype in (
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                delta = _get_val(event, "delta") or ""
                ui.thinking(delta)
                reasoning_parts.append(delta)

            elif etype == "response.output_item.added":
                item = _get_val(event, "item")
                if _get_val(item, "type") == "function_call":
                    key = str(
                        _get_val(item, "id")
                        or _get_val(item, "call_id")
                        or _get_val(event, "output_index")
                    )
                    tool_calls[key] = {
                        "id": _get_val(item, "call_id") or _get_val(item, "id") or key,
                        "name": _get_val(item, "name") or "",
                        "arguments": _get_val(item, "arguments") or "",
                    }
                    ui.tool_progress(
                        ", ".join(t["name"] or "tool" for t in tool_calls.values()),
                        sum(len(t["arguments"]) for t in tool_calls.values()) / 1024.0,
                    )

            elif etype == "response.function_call_arguments.delta":
                key = _tool_key(event)
                if key not in tool_calls:
                    tool_calls[key] = {
                        "id": _get_val(event, "item_id") or key,
                        "name": "",
                        "arguments": "",
                    }
                tool_calls[key]["arguments"] += _get_val(event, "delta") or ""
                ui.tool_progress(
                    ", ".join(t["name"] or "tool" for t in tool_calls.values()),
                    sum(len(t["arguments"]) for t in tool_calls.values()) / 1024.0,
                )

            elif etype == "response.function_call_arguments.done":
                key = _tool_key(event)
                if key in tool_calls:
                    name = _get_val(event, "name")
                    args = _get_val(event, "arguments")
                    if name:
                        tool_calls[key]["name"] = name
                    if args and not tool_calls[key]["arguments"]:
                        tool_calls[key]["arguments"] = args

            elif etype == "response.completed":
                resp = _get_val(event, "response")
                usage_obj = _get_val(resp, "usage") if resp is not None else None

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
    def _tools_to_responses(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict):
                converted.append(
                    {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            elif isinstance(tool, dict) and tool.get("type") == "function":
                converted.append(tool)
        return converted or None

    @staticmethod
    def _messages_to_responses_input(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        instructions_parts: list[str] = []
        items: list[dict[str, Any]] = []
        for msg in messages:
            role = (msg.get("role") or "").lower()
            content = msg.get("content") or ""
            if role == "system":
                if content:
                    instructions_parts.append(str(content))
                continue
            if role == "user":
                items.append({"role": "user", "content": content})
                continue
            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if content:
                    items.append({"role": "assistant", "content": content})
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc, dict) else {}
                    if not isinstance(fn, dict):
                        fn = {}
                    call_id = tc.get("id") if isinstance(tc, dict) else None
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id or "call_unknown",
                            "name": fn.get("name") or "unknown",
                            "arguments": fn.get("arguments") or "{}",
                        }
                    )
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id") or "call_unknown",
                        "output": str(content),
                    }
                )
        return "\n\n".join(instructions_parts), items

    # ------------------------------------------------------------------
    # Chat Completions (compat hosts)
    # ------------------------------------------------------------------

    def _complete_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        reasoning_effort: str | None,
        stream_handler: StreamHandler | None,
        count_usage: Callable[..., Any] | None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            if (self.name or "").lower() == "openrouter":
                kwargs["include_reasoning"] = True

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("stream_options", None)
            kwargs.pop("reasoning_effort", None)
            stream = self.client.chat.completions.create(**kwargs)

        ui = _StreamUI(stream_handler)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[Any, dict[str, Any]] = {}
        usage_obj: Any = None

        for chunk in stream:
            current_usage = _get_val(chunk, "usage")
            if current_usage:
                usage_obj = current_usage

            choices = _get_val(chunk, "choices")
            if not choices:
                continue
            delta = _get_val(choices[0], "delta")
            if not delta:
                continue

            reasoning = _get_val(delta, "reasoning_content") or _get_val(
                delta, "reasoning"
            )
            if reasoning:
                ui.thinking(reasoning)
                reasoning_parts.append(reasoning)

            content = _get_val(delta, "content")
            if content:
                ui.content(content)
                content_parts.append(content)

            streamed_tools = _get_val(delta, "tool_calls")
            if streamed_tools:
                for tc in streamed_tools:
                    idx = _get_val(tc, "index")
                    tc_id = _get_val(tc, "id")
                    tc_function = _get_val(tc, "function")
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_id:
                        tool_calls_map[idx]["id"] = tc_id
                    if tc_function:
                        fn_name = _get_val(tc_function, "name")
                        fn_args = _get_val(tc_function, "arguments")
                        if fn_name:
                            tool_calls_map[idx]["function"]["name"] += fn_name
                        if fn_args:
                            tool_calls_map[idx]["function"]["arguments"] += fn_args
                    total_arg_bytes = sum(
                        len(row["function"]["arguments"])
                        for row in tool_calls_map.values()
                    )
                    names = ", ".join(
                        row["function"]["name"] or "tool"
                        for row in tool_calls_map.values()
                    )
                    ui.tool_progress(names, total_arg_bytes / 1024.0)

        ui.finish()
        usage_obj = _normalize_usage(usage_obj)
        if usage_obj is None and count_usage is not None:
            usage_obj = count_usage(messages, "".join(content_parts))

        final_tool_calls = [
            ToolCall(
                id=row["id"],
                function=ToolCallFunction(
                    name=row["function"]["name"],
                    arguments=row["function"]["arguments"],
                ),
            )
            for _, row in sorted(tool_calls_map.items(), key=lambda kv: str(kv[0]))
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

    def list_models(self) -> list[ModelInfo]:
        raw = self._fetch_raw_models()
        results: list[ModelInfo] = []
        seen: set[str] = set()
        if raw:
            for m in raw:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                if not mid or str(mid) in seen:
                    continue
                seen.add(str(mid))
                results.append(ModelInfo(id=str(mid), raw=m))
            return results

        response = self.client.models.list()
        for m in getattr(response, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid or str(mid) in seen:
                continue
            seen.add(str(mid))
            results.append(ModelInfo(id=str(mid), raw=m))
        return results

    def _fetch_raw_models(self) -> list[dict[str, Any]]:
        try:
            url = f"{self.base_url}/models"
            headers = {"User-Agent": "pi-python/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = Request(url, headers=headers)
            with urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict):
                    return data.get("data", []) or []
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []
