"""Slash-command handlers for the PI agent REPL."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from llm import Agent

CommandResult = Union[bool, str, None]


class Commands:
    """All user-facing slash commands live here."""

    ALIASES: Dict[str, str] = {
        "exit": "/exit",
        "quit": "/exit",
        "help": "/help",
    }

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.commands = self.get_methods()
        self.agent.console.set_slash_commands(self.command_names())

    def command_names(self) -> List[str]:
        names: List[str] = []
        for entry in self.commands:
            names.extend(entry.keys())
        # Include bare aliases for completion
        names.extend(["exit", "quit"])
        # stable unique order
        seen: set[str] = set()
        ordered: List[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        return ordered

    def command_help_rows(self) -> List[tuple[str, str]]:
        rows: List[tuple[str, str]] = []
        for entry in self.commands:
            for name, fn in entry.items():
                doc = (inspect.getdoc(fn) or "").strip().split("\n")[0]
                rows.append((name, doc or name))
        rows.append(("exit / quit", "End the session"))
        return rows

    def router(self, command: str) -> CommandResult:
        raw = (command or "").strip()
        if not raw:
            return None

        key = raw.lower()
        key = self.ALIASES.get(key, key)

        if not key.startswith("/"):
            return None

        for entry in self.commands:
            handler = entry.get(key)
            if handler is not None:
                return handler()
        return None

    def get_methods(self) -> List[Dict[str, Callable[..., Any]]]:
        """Auto-register public methods as /<name> commands."""
        methods: List[Dict[str, Callable[..., Any]]] = []
        skip = {"router", "command_names", "command_help_rows", "get_methods"}

        for attr_name in dir(self):
            if attr_name.startswith("_") or attr_name in skip:
                continue
            if attr_name.startswith("get"):
                continue

            attr = getattr(self, attr_name)
            if inspect.ismethod(attr):
                methods.append({f"/{attr_name}": attr})

        return methods

    def help(self) -> bool:
        """Show this help"""
        self.agent.console.print_help(self.command_help_rows())
        return True

    def clear(self) -> bool:
        """Clear the screen"""
        agent = self.agent
        agent.console.clear_screen()
        agent.console.print_welcome(
            agent.console._session_title,
            agent.console._session_workspace,
            is_auth=agent.config.api_key,
        )
        return True

    def quiet(self) -> bool:
        """Collapse tool output to one-liners"""
        console = self.agent.console
        console.quiet = True
        console.verbose = False
        console.print_system_message("Tool output collapsed (quiet mode).", title="Mode")
        return True

    def verbose(self) -> bool:
        """Show full tool output"""
        console = self.agent.console
        console.quiet = False
        console.verbose = True
        console.print_system_message("Full tool output enabled (verbose mode).", title="Mode")
        return True

    def copy(self) -> bool:
        """Copy last assistant reply to clipboard"""
        console = self.agent.console
        if not console._last_assistant_message:
            console.print_error("No assistant message to copy yet.", title="Copy")
        elif console._copy_to_clipboard(console._last_assistant_message):
            console.print_system_message(
                "Last assistant reply copied to clipboard.", title="Copy"
            )
        else:
            console.print_error("Could not access the system clipboard.", title="Copy")
        return True

    def resume(self) -> bool:
        """Resume a previous session"""
        agent = self.agent
        old_sessions = agent.memory.load_old_sessions()
        if not old_sessions:
            agent.console.print_system_message("No previous sessions found.", "Resume")
            return True
        selected_session = agent.console.interactive_select(old_sessions)
        agent.console.clear_screen()
        old_chats = agent.memory.load_session_chat(
            selected_session.history_path,
            system_prompt=agent.prompt.raw_system_prompt,
        )
        agent.memory.session = selected_session
        agent.console.print_welcome(
            selected_session.title,
            str(selected_session.workspace),
            is_auth=agent.config.api_key,
        )
        agent.console.print_chat_history(old_chats)
        agent.console.print_system_message(f"Resumed session: {selected_session.title}")
        return True

    def login(self) -> bool:
        """Authenticate provider API key"""
        agent = self.agent
        api_key = agent.console.get_api_key(agent.config.provider)
        if not api_key:
            return True

        try:
            auth_root = agent.memory.root / "auth.json"
            data = agent.memory.read_from_json(auth_root) or {}
            credentials = data.get("credentials") or {}
            credentials["api_key"] = api_key
            data["credentials"] = credentials
            agent.memory.write_to_json(auth_root, data)
            agent.config.api_key = api_key
            agent.client = agent.create_model()
        except Exception as e:
            agent.console.print_error(f"Failed to save credentials: {e}", title="Auth Error")
        return True

    def exit(self) -> str:
        """End the session"""
        self.agent.console.print_system_message("Goodbye!", "Exit")
        return "exit"
