"""Enhanced console UI for the PI agent (Rich + prompt_toolkit)."""

from __future__ import annotations

from contextlib import contextmanager
import json
import re
import subprocess
import sys
import time
from typing import Any, List, Optional

from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from prompt_toolkit import Application, prompt as PtPrompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.styles import Style as PtStyle

from models import Session


# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------

THEME = {
    "brand": "bold white on blue",
    "user": "cyan",
    "user_border": "green",
    "user_title": "bold green",
    "assistant": "white",
    "assistant_border": "blue",
    "assistant_title": "bold blue",
    "system_border": "blue",
    "tool": "yellow",
    "tool_border": "yellow",
    "tool_title": "bold yellow",
    "tool_result": "green",
    "error": "red",
    "error_border": "red",
    "error_title": "bold red",
    "muted": "dim",
    "accent": "cyan",
    "warn": "bold yellow",
    "ok": "bold green",
    "deny": "bold red",
}

TOOL_PREVIEW_CHARS = 400
TOOL_PREVIEW_LINES = 12
TOOL_ARGS_PREVIEW = 160
HISTORY_TOOL_CHARS = 200
HISTORY_TOOL_LINES = 4
FENCE_RE = re.compile(r"```(\w+)?\n", re.MULTILINE)


class TimedStatus:
    """Stable Rich Spinner with elapsed time (no custom frame/color thrashing)."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._start = time.time()
        # Fixed style — avoid dim/markup rebuilds that flicker on Windows consoles
        self._spinner = Spinner("dots", style="cyan")

    def __rich_console__(self, console, options):
        elapsed = int(time.time() - self._start)
        if "ESC" in self.message:
            self._spinner.text = f"{self.message} [{elapsed}s]"
        else:
            self._spinner.text = f"{self.message} [{elapsed}s] - ESC to stop"
        yield self._spinner


class ConsoleUI:
    """Enhanced console interface for the pi agent."""

    def __init__(self, history_file: str = ".pi_history"):
        self.console = Console()
        self.history_file = history_file
        self._setup_history()
        self._current_live: Optional[Live] = None
        self._status: Optional[TimedStatus] = None
        self.quiet: bool = False
        self.verbose: bool = False
        self._last_assistant_message: str = ""
        self._session_title: str = " "
        self._session_workspace: str = ""

    def _setup_history(self) -> None:
        history_path = Path(self.history_file)
        if not history_path.exists():
            history_path.touch()

    @contextmanager
    def _pause_loading(self):
        """
        Temporarily stop the Live spinner so other prints don't fight it
        (Windows consoles otherwise flicker / bleed colors).
        """
        status = self._status
        live = self._current_live
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass
            self._current_live = None
        try:
            yield
        finally:
            # Resume only if we still want loading and nothing else took over
            if status is not None and self._current_live is None and self._status is status:
                self._current_live = Live(
                    status,
                    console=self.console,
                    refresh_per_second=8,
                    transient=False,
                )
                try:
                    self._current_live.start()
                except Exception:
                    self._current_live = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str, max_chars: int, max_lines: int) -> tuple[str, bool]:
        """Return (display_text, was_truncated)."""
        if not text:
            return text, False
        lines = text.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
            text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars].rstrip() + "..."
            truncated = True
        elif truncated:
            text = "\n".join(lines) + "\n..."
        return text, truncated

    def _pretty_args(self, arguments: str) -> tuple[str, bool]:
        try:
            parsed = json.loads(arguments)
            pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            pretty = arguments
        if self.verbose:
            return pretty, False
        return self._truncate(pretty, TOOL_ARGS_PREVIEW, 6)

    def _detect_code_language(self, message: str) -> Optional[str]:
        match = FENCE_RE.search(message or "")
        if match and match.group(1):
            return match.group(1).lower()
        return None

    def _copy_to_clipboard(self, text: str) -> bool:
        if not text:
            return False
        try:
            if sys.platform == "win32":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.run(
                    ["clip"],
                    input=text.encode("utf-16"),
                    check=True,
                    creationflags=flags,
                )
                return True
            if sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
                return True
            for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
                try:
                    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
                    return True
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        except Exception:
            return False
        return False

    def _print_help(self) -> None:
        help_body = Text()
        help_body.append("Commands\n", style="bold cyan")
        rows = [
            ("/help", "Show this help"),
            ("/clear", "Clear the screen"),
            ("/quiet", "Collapse tool output to one-liners"),
            ("/verbose", "Show full tool output"),
            ("/copy", "Copy last assistant reply to clipboard"),
            ("/resume", "Resume a previous session"),
            ("/login", "Authenticate provider api key"),
            ("exit / quit", "End the session"),
        ]
        for cmd, desc in rows:
            help_body.append(f"  {cmd:<14}", style="green")
            help_body.append(f"{desc}\n", style="dim")

        help_body.append("\nKeys\n", style="bold cyan")
        keys = [
            ("Enter", "Submit task"),
            ("Ctrl+J", "Insert newline (multiline)"),
            ("Esc Esc+Enter", "Insert newline (alt)"),
            ("ESC (during run)", "Stop current agent turn"),
            ("Ctrl+C (at prompt)", "Exit"),
            ("Up/Down", "History / session picker"),
        ]
        for key, desc in keys:
            help_body.append(f"  {key:<18}", style="yellow")
            help_body.append(f"{desc}\n", style="dim")

        mode = "quiet" if self.quiet else ("verbose" if self.verbose else "normal")
        help_body.append(f"\nTool output mode: ", style="dim")
        help_body.append(mode, style="bold cyan")

        self.console.print(
            Panel(help_body, title="[bold cyan]Help[/bold cyan]", border_style=THEME["accent"], box=box.ROUNDED)
        )

    def _handle_local_command(self, raw: str) -> bool:
        """
        Handle UI-only slash commands.
        Returns True if the input was consumed (caller should re-prompt).
        """
        cmd = raw.strip().lower()
        if cmd in ("/help", "help"):
            self._print_help()
            return True
        if cmd == "/clear":
            self.clear_screen()
            return True
        if cmd == "/quiet":
            self.quiet = True
            self.verbose = False
            self.print_system_message("Tool output collapsed (quiet mode).", title="Mode")
            return True
        if cmd == "/verbose":
            self.quiet = False
            self.verbose = True
            self.print_system_message("Full tool output enabled (verbose mode).", title="Mode")
            return True
        if cmd == "/copy":
            if not self._last_assistant_message:
                self.print_error("No assistant message to copy yet.", title="Copy")
            elif self._copy_to_clipboard(self._last_assistant_message):
                self.print_system_message("Last assistant reply copied to clipboard.", title="Copy")
            else:
                self.print_error("Could not access the system clipboard.", title="Copy")
            return True
        return False

    def _get_completer(self) -> Completer:
        commands = [
            "/resume",
            "/help",
            "/clear",
            "/quiet",
            "/verbose",
            "/copy",
            "exit",
            "quit",
        ]

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                word = document.get_word_before_cursor()
                text = document.text_before_cursor
                # Complete slash commands from start of line
                if text.lstrip().startswith("/") or not text.strip():
                    prefix = text.lstrip()
                    for cmd in commands:
                        if cmd.startswith(prefix):
                            yield Completion(cmd, start_position=-len(prefix))
                else:
                    for cmd in commands:
                        if cmd.startswith(word):
                            yield Completion(cmd, start_position=-len(word))

        return SlashCompleter()

    # ------------------------------------------------------------------
    # Loading / status
    # ------------------------------------------------------------------

    def start_loading(self, message: str = "Thinking..."):
        """Start loading spinner if not already running."""
        if self._current_live is None:
            self._status = TimedStatus(message)
            self._current_live = Live(
                self._status,
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self._current_live.start()

    def stop_loading(self):
        """Stop loading spinner if active."""
        if self._current_live is not None:
            try:
                self._current_live.stop()
            except Exception:
                pass
            self._current_live = None
        self._status = None

    @contextmanager
    def print_loading(self, message: str = "Thinking..."):
        """Show loading spinner safely without overlapping interactive prompts."""
        self.start_loading(message)
        try:
            yield self._current_live
        finally:
            self.stop_loading()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _erase_prompt_echo(self, text: str) -> None:
        """Remove the leftover 'Task > ...' line(s) so the User block isn't a duplicate."""
        line_count = max(1, text.count("\n") + 1)
        try:
            for _ in range(line_count):
                sys.stdout.write("\033[1A\033[2K")
            sys.stdout.flush()
        except Exception:
            pass

    def get_user_input(self, prompt: str = "Enter your task") -> str:
        """Get user input with history, multiline, local slash commands."""
        self.stop_loading()
        bindings = KeyBindings()

        @bindings.add("c-j")
        def _newline_ctrl_j(event) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "enter")
        def _newline_esc_enter(event) -> None:
            event.current_buffer.insert_text("\n")

        while True:
            try:
                user_input = PtPrompt(
                    f"{prompt} > ",
                    history=FileHistory(self.history_file),
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=self._get_completer(),
                    complete_while_typing=True,
                    key_bindings=bindings,
                    multiline=False,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.print_error(f"Advanced input failed: {str(e)}. Using basic input.")
                user_input = input(f"{prompt} > ")

            text = (user_input or "").strip()
            if not text:
                return ""
            if self._handle_local_command(text):
                continue
            # Drop the raw prompt echo; caller prints a single User block
            self._erase_prompt_echo(f"{prompt} > {user_input}")
            return text

    def interactive_select(
        self,
        items: List[Session],
        title: str = "Select a session",
        prompt: str = "Enter number",
    ) -> Session:
        """Arrow-key session picker (Up/Down, Enter). Number entry still works as fallback."""
        self.stop_loading()
        if not items:
            raise ValueError("No items to select from")

        selected = {"index": 0}
        cancelled = {"flag": False}

        def _label(i: int, session: Session) -> str:
            title_text = (session.title or session.id)[:72]
            ws = str(getattr(session, "workspace", ""))
            ws_short = ws if len(ws) <= 40 else "..." + ws[-39:]
            marker = ">" if i == selected["index"] else " "
            return f"{marker} {i + 1}. {title_text}  ({ws_short})"

        def get_text() -> FormattedText:
            fragments: list[tuple[str, str]] = [
                ("bold", f"{title}\n"),
                ("class:muted", "Up/Down move | Enter select | Esc cancel | or type a number\n\n"),
            ]
            for i, session in enumerate(items):
                style = "class:selected" if i == selected["index"] else ""
                fragments.append((style, _label(i, session) + "\n"))
            return FormattedText(fragments)

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            selected["index"] = (selected["index"] - 1) % len(items)

        @kb.add("down")
        def _down(event) -> None:
            selected["index"] = (selected["index"] + 1) % len(items)

        @kb.add("enter")
        def _enter(event) -> None:
            event.app.exit(result=items[selected["index"]])

        @kb.add("escape")
        def _esc(event) -> None:
            cancelled["flag"] = True
            event.app.exit(result=None)

        for n in range(1, min(len(items), 9) + 1):
            @kb.add(str(n))
            def _num(event, n=n) -> None:
                event.app.exit(result=items[n - 1])

        style = PtStyle.from_dict({
            "selected": "bold reverse cyan",
            "muted": "italic #888888",
        })

        try:
            app: Application[Optional[Session]] = Application(
                layout=Layout(Window(FormattedTextControl(get_text), always_hide_cursor=True)),
                key_bindings=kb,
                style=style,
                full_screen=False,
            )
            result = app.run()
            if result is not None:
                return result
            if cancelled["flag"]:
                # Fall back to numeric prompt rather than crashing resume flow
                self.print_system_message("Picker cancelled — enter a session number.", title="Resume")
        except Exception:
            pass

        # Numeric fallback (also used if Application fails)
        self.console.print(
            Panel(Text(title, style="bold cyan"), border_style=THEME["accent"], box=box.ROUNDED)
        )
        for i, item in enumerate(items, 1):
            self.console.print(f"[bold cyan]{i}.[/bold cyan] {item.title}")
        self.console.print()

        while True:
            try:
                from rich.prompt import Prompt

                choice = Prompt.ask(f"{prompt}", console=self.console)
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    return items[idx]
                self.print_error(f"Invalid selection. Choose a number between 1 and {len(items)}.")
            except ValueError:
                self.print_error("Please enter a valid number.")

    # ------------------------------------------------------------------
    # Message printers
    # ------------------------------------------------------------------

    def print_system_message(self, message: str, title: str = "System"):
        self.console.print(
            Panel(
                Text(message, style="white"),
                title=title,
                border_style=THEME["system_border"],
                box=box.DOUBLE,
            )
        )

    def print_user_message(self, message: str):
        self.console.print(
            Panel(
                Text(message, style=THEME["user"]),
                title=f"[{THEME['user_title']}]User[/{THEME['user_title']}]",
                border_style=THEME["user_border"],
                box=box.ROUNDED,
            )
        )

    def print_assistant_message(self, message: str, is_code: bool = False):
        """Print assistant response; skips empty content; detects fenced languages."""
        content = message or ""
        if content.strip():
            self._last_assistant_message = content

        if not content.strip():
            self.console.print(
                Text("  (assistant returned no text)", style=THEME["muted"])
            )
            return

        title = f"[{THEME['assistant_title']}]Assistant[/{THEME['assistant_title']}]"
        border = THEME["assistant_border"]

        if is_code:
            lang = self._detect_code_language(content) or "python"
            try:
                syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title=title, border_style=border, box=box.ROUNDED))
                return
            except Exception:
                pass

        # Prefer markdown; if a single fenced block dominates, still fine via Markdown
        try:
            self.console.print(
                Panel(Markdown(content), title=title, border_style=border, box=box.ROUNDED)
            )
        except Exception:
            self.console.print(
                Panel(
                    Text(content, style=THEME["assistant"]),
                    title=title,
                    border_style=border,
                    box=box.ROUNDED,
                )
            )

    def print_tool_call(self, tool_name: str, arguments: str):
        with self._pause_loading():
            if self.quiet:
                summary = arguments.replace("\n", " ")
                if len(summary) > 80:
                    summary = summary[:77] + "..."
                line = Text()
                line.append("> ", style=THEME["tool"])
                line.append(tool_name, style="cyan")
                line.append(" ")
                line.append(summary, style=THEME["muted"])
                self.console.print(line)
                return

            args_display, truncated = self._pretty_args(arguments)
            self.console.print(
                f"[bold {THEME['tool']}]>[/] Calling [cyan]{tool_name}[/cyan]"
            )
            if arguments:
                body: RenderableType = Text(args_display, style="white")
                if truncated:
                    body = Group(
                        body,
                        Text("... truncated - /verbose for full output", style=THEME["muted"]),
                    )
                self.console.print(
                    Panel(
                        body,
                        title=f"[{THEME['tool_title']}]args[/{THEME['tool_title']}]",
                        border_style=THEME["tool_border"],
                        box=box.MINIMAL,
                    )
                )

    def print_tool_result(self, result: str):
        with self._pause_loading():
            text = result or ""
            if self.quiet:
                preview, _ = self._truncate(text.replace("\n", " "), 100, 1)
                self.console.print(Text(f"  -> {preview}", style=THEME["muted"]))
                return

            if self.verbose:
                body = text
                truncated = False
            else:
                body, truncated = self._truncate(text, TOOL_PREVIEW_CHARS, TOOL_PREVIEW_LINES)

            renderable: RenderableType
            try:
                renderable = Text(body, style=THEME["tool_result"])
                if truncated:
                    renderable = Group(
                        renderable,
                        Text("... truncated - /verbose for full output", style=THEME["muted"]),
                    )
            except Exception:
                renderable = Text(body, style=THEME["tool_result"])

            self.console.print(
                Panel(
                    renderable,
                    title=f"[{THEME['tool_title']}]Tool Result[/{THEME['tool_title']}]",
                    border_style=THEME["tool_border"],
                    box=box.ROUNDED,
                )
            )

    def print_error(self, error: str, title: str = "Error"):
        with self._pause_loading():
            self.console.print(
                Panel(
                    Text(error, style=THEME["error"]),
                    title=f"[{THEME['error_title']}]{title}[/{THEME['error_title']}]",
                    border_style=THEME["error_border"],
                    box=box.HEAVY,
                )
            )

    def print_welcome(
        self,
        title: str = " ",
        ws_path: str = "",
        is_auth: str | None = None,
    ):
        if title and title.strip():
            self._session_title = title
        if ws_path:
            self._session_workspace = ws_path

        welcome_text = Text()
        welcome_text.append("\n")
        welcome_text.append("  PI - Python Agent Harness  ", style=THEME["brand"])

        if self._session_title.strip() and self._session_workspace:
            welcome_text.append("\n")
            welcome_text.append(
                f"Title: {self._session_title} \n Workspace : {self._session_workspace} ",
                style="bold cyan",
            )

        welcome_text.append("\n")
        welcome_text.append("  Type 'exit' or 'quit' to end - /help for commands  ", style=THEME["muted"])
        welcome_text.append("\n")
        welcome_text.append("  Press ESC to stop the current agent turn  ", style=THEME["muted"])
        welcome_text.append("\n")
        welcome_text.append("  Ctrl+J for multiline input  ", style=THEME["muted"])
        welcome_text.append("\n")

        self.console.print(Panel(welcome_text, box=box.DOUBLE))

        if not is_auth:
            auth_warning = Text()
            auth_warning.append("⚠️  No API Key / Credentials Found\n", style=THEME["warn"])
            auth_warning.append(
                "You are currently unauthenticated. Run ", style="white"
            )
            auth_warning.append("/login", style="bold cyan")
            auth_warning.append(
                " to configure your provider API key before sending tasks.", style="white"
            )

            self.console.print(
                Panel(
                    auth_warning,
                    title=f"[{THEME['error_title']}]Authentication Required[/{THEME['error_title']}]",
                    border_style=THEME["error_border"],
                    box=box.ROUNDED,
                )
            )

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def confirm_permission(self, action_details: str, title: str = "Permission Required") -> bool:
        was_loading = self._current_live is not None
        if was_loading:
            self.stop_loading()

        self.console.print(
            Panel(
                Text(action_details, style=THEME["warn"]),
                title=f"[bold red]WARNING: {title}[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        res = Confirm.ask("Do you allow this action?", console=self.console, default=False)

        if was_loading:
            self.start_loading("Processing your request...")
        return res

    def confirm_permission_extended(self, tool_name: str, target: str, action_details: str) -> str:
        """
        Single-key permission prompt.
        Returns: 'y' | 'a' | 'f' | 'all' | 'n'  (same contract as before).
        Keys: y, a, f, l (all), n — Enter not required.
        """
        was_loading = self._current_live is not None
        if was_loading:
            self.stop_loading()

        clean_target = target[:60] + "..." if len(target) > 60 else target
        risk = "high" if tool_name in ("bash", "write", "edit") else "normal"
        risk_style = "red" if risk == "high" else "yellow"

        prompt_text = (
            f"[bold yellow]{action_details}[/bold yellow]\n\n"
            f"[cyan]Press a key:[/cyan]\n"
            f"  [{THEME['ok']}][y][/{THEME['ok']}]   Allow Once\n"
            f"  [{THEME['ok']}][a][/{THEME['ok']}]   Always Allow Tool [yellow]'{tool_name}'[/yellow]\n"
            f"  [{THEME['ok']}][f][/{THEME['ok']}]   Always Allow Target [yellow]'{clean_target}'[/yellow]\n"
            f"  [{THEME['ok']}][l][/{THEME['ok']}]   Always Allow ALL tools\n"
            f"  [{THEME['deny']}][n][/{THEME['deny']}]   Deny"
        )

        self.console.print(
            Panel(
                Text.from_markup(prompt_text),
                title=f"[bold {risk_style}]WARNING: Permission ({tool_name})[/bold {risk_style}]",
                border_style=risk_style,
                box=box.ROUNDED,
            )
        )

        choice_map = {
            "y": "y",
            "a": "a",
            "f": "f",
            "l": "all",  # single-key for "all"
            "n": "n",
        }
        result = {"value": "n"}

        kb = KeyBindings()

        for key, value in choice_map.items():
            @kb.add(key)
            @kb.add(key.upper())
            def _pick(event, value=value) -> None:
                result["value"] = value
                event.app.exit()

        # Still accept typing "all" via 'a'+'l' path: dedicated binding already maps l→all
        @kb.add("escape")
        def _deny(event) -> None:
            result["value"] = "n"
            event.app.exit()

        try:
            self.console.print(Text("Choice > ", style="bold"), end="")
            app: Application[None] = Application(
                layout=Layout(Window(FormattedTextControl(lambda: ""), height=1)),
                key_bindings=kb,
                full_screen=False,
            )
            app.run()
            self.console.print(f"[bold]{result['value']}[/bold]")
        except Exception:
            from rich.prompt import Prompt

            typed = Prompt.ask(
                "Permission choice",
                choices=["y", "a", "f", "all", "n", "l"],
                default="n",
                console=self.console,
            ).lower().strip()
            result["value"] = "all" if typed == "l" else typed

        if was_loading:
            self.start_loading("Processing your request...")

        return result["value"]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def print_usage_summary(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        """Quiet one-line session token/cost summary."""
        line = (
            f"tokens {total_tokens:,} "
            f"(in {prompt_tokens:,} / out {completion_tokens:,}) "
            f"~ ${estimated_cost_usd:.4f}"
        )
        self.console.print(Text(line, style=THEME["muted"]))

    def print_separator(self):
        self.console.print("─" * self.console.width, style=THEME["muted"])

    def clear_screen(self):
        self.console.clear()
        # self.print_welcome(self._session_title, self._session_workspace)

    def print_code_block(self, code: str, language: str = "python"):
        try:
            self.console.print(Syntax(code, language, theme="monokai", line_numbers=True))
        except Exception:
            self.console.print(f"[dim]Code block ({language}):[/dim]")
            self.console.print(code)

    def print_chat_history(self, messages: List[Any]) -> None:
        """Replay history with system prompt collapsed and tool results trimmed."""
        if not messages:
            self.console.print(
                Panel(
                    Text("No chat history found.", style="dim italic"),
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )
            return

        self.console.print(
            Panel(
                Text("Chat History", style="bold cyan"),
                border_style=THEME["accent"],
                box=box.ROUNDED,
            )
        )

        for msg in messages:
            role_val = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            role = role_val.lower()
            content = getattr(msg, "content", "") or ""

            if role == "user":
                self.print_user_message(content)
            elif role in ("assistant", "model"):
                # Skip empty assistant turns that only carried tool_calls
                tool_calls = getattr(msg, "tool_calls", None)
                if not content.strip() and tool_calls:
                    n = len(tool_calls) if isinstance(tool_calls, list) else 1
                    self.console.print(
                        Text(f"  (assistant requested {n} tool call(s))", style=THEME["muted"])
                    )
                else:
                    self.print_assistant_message(content)
            elif role == "system":
                preview, _ = self._truncate(content.replace("\n", " "), 120, 1)
                self.console.print(
                    Panel(
                        Text(f"{preview}\n\n(system prompt collapsed)", style=THEME["muted"]),
                        title="System Instruction",
                        border_style="dim",
                        box=box.ROUNDED,
                    )
                )
            elif role == "tool":
                name = getattr(msg, "name", None) or "tool"
                body, truncated = self._truncate(content, HISTORY_TOOL_CHARS, HISTORY_TOOL_LINES)
                label = f"Tool:{name}"
                if truncated:
                    body = body.rstrip(".") + "\n... (collapsed)"
                self.console.print(
                    Panel(
                        Text(body, style=THEME["tool_result"]),
                        title=f"[dim]{label}[/dim]",
                        border_style="dim",
                        box=box.MINIMAL,
                    )
                )

        self.print_separator()
        
    def get_api_key(
        self,
        provider: str = "Provider",
        env_var_hint: Optional[str] = None,
    ) -> str:
        """
        Securely prompt the user to enter an API key with masked input.
        Returns the entered API key string.
        """
        self.stop_loading()
        title_text = f"Authentication - {provider}"
        hint = f" (or set {env_var_hint} in environment)" if env_var_hint else ""

        self.console.print(
            Panel(
                Text(f"Please enter your API key for {provider}{hint}:", style="bold cyan"),
                title=title_text,
                border_style=THEME["accent"],
                box=box.ROUNDED,
            )
        )

        while True:
            try:
                # Mask user input securely
                api_key = PtPrompt(
                    f"{provider} API Key > ",
                    is_password=True,
                ).strip()
            except KeyboardInterrupt:
                raise
            except Exception:
                # Fallback to rich Prompt password masking if PtPrompt fails
                from rich.prompt import Prompt
                api_key = Prompt.ask(
                    f"{provider} API Key",
                    password=True,
                    console=self.console,
                ).strip()

            if api_key:
                self.print_system_message(
                    f"API key for {provider} set successfully.", title="Auth Success"
                )
                return api_key

            self.print_error("API key cannot be empty. Please try again.", title="Auth Error")

# Global console instance
console_ui = ConsoleUI()


def get_console() -> ConsoleUI:
    """Get the global console instance."""
    return console_ui
