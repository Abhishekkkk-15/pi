"""Shared streaming UI pairing for provider implementations."""

from __future__ import annotations

from providers.base import StreamHandler


class StreamUI:
    def __init__(self, handler: StreamHandler | None) -> None:
        self.handler = handler
        self.thinking_open = False
        self.content_open = False
        self.loading_active = False

    def thinking(self, text: str) -> None:
        if not text:
            return
        if self.handler is not None:
            self.stop_load()
            if not self.thinking_open:
                self.handler.thinking_start()
                self.thinking_open = True
            self.handler.thinking_chunk(text)

    def content(self, text: str) -> None:
        if not text:
            return
        if self.handler is not None:
            self.close_thinking()
            self.stop_load()
            if not self.content_open:
                self.handler.content_start()
                self.content_open = True
            self.handler.content_chunk(text)

    def tool_progress(self, names: str, kb: float) -> None:
        self.close_thinking()
        self.close_content()
        if self.handler is not None:
            self.handler.tool_args_progress(names, kb)
            self.loading_active = True

    def close_thinking(self) -> None:
        if self.thinking_open and self.handler is not None:
            self.handler.thinking_end()
        self.thinking_open = False

    def close_content(self) -> None:
        if self.content_open and self.handler is not None:
            self.handler.content_end()
        self.content_open = False

    def stop_load(self) -> None:
        if self.loading_active and self.handler is not None:
            self.handler.stop_loading()
        self.loading_active = False

    def finish(self) -> None:
        self.close_thinking()
        self.close_content()
        self.stop_load()
