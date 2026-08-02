"""Enhanced console UI for the PI agent (Rich + prompt_toolkit).

Visual language is intentionally minimal, in the spirit of modern coding CLIs:
gutter markers instead of heavy panels, dim secondary text, one accent color.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
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
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from prompt_toolkit import Application, prompt as PtPrompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.styles import Style as PtStyle

from models import Session


# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------

THEMES: dict[str, dict[str, str]] = {
    "ember": {
        "accent": "#d98452",
        "accent_dim": "#a8613a",
        "ink": "#e6e6e6",
        "subtle": "#8a8a8a",
        "warn": "yellow",
        "error": "red",
        "ok": "green",
    },
    "mono": {
        "accent": "#f2f2f2",
        "accent_dim": "#6f6f6f",
        "ink": "#dcdcdc",
        "subtle": "#7d7d7d",
        "warn": "#c8c8c8",
        "error": "#ff6b6b",
        "ok": "#c8c8c8",
    },
    "ocean": {
        "accent": "#4bb3d4",
        "accent_dim": "#2b6f88",
        "ink": "#e3edf2",
        "subtle": "#7d97a3",
        "warn": "#e6c07b",
        "error": "#ef6b73",
        "ok": "#5fc9a0",
    },
    "forest": {
        "accent": "#7cc47f",
        "accent_dim": "#4a7a4d",
        "ink": "#e6efe4",
        "subtle": "#849586",
        "warn": "#d8bd6a",
        "error": "#e0736b",
        "ok": "#7cc47f",
    },
    "violet": {
        "accent": "#b48ee8",
        "accent_dim": "#70539b",
        "ink": "#ece7f5",
        "subtle": "#93899f",
        "warn": "#e2c169",
        "error": "#ec6f8b",
        "ok": "#8fd6a8",
    },
    "contrast": {
        "accent": "bright_yellow",
        "accent_dim": "yellow",
        "ink": "bright_white",
        "subtle": "white",
        "warn": "bright_yellow",
        "error": "bright_red",
        "ok": "bright_green",
    },
}

DEFAULT_THEME = "ember"

# Live style globals — rebound by apply_theme(), read at call time everywhere
ACCENT = THEMES[DEFAULT_THEME]["accent"]
ACCENT_DIM = THEMES[DEFAULT_THEME]["accent_dim"]
INK = THEMES[DEFAULT_THEME]["ink"]
SUBTLE = THEMES[DEFAULT_THEME]["subtle"]

THEME: dict[str, str] = {}
_active_theme = DEFAULT_THEME


def apply_theme(name: str) -> str:
    """Rebind palette globals in place. Returns the theme actually applied."""
    global ACCENT, ACCENT_DIM, INK, SUBTLE, _active_theme

    key = (name or "").strip().lower()
    if key not in THEMES:
        key = DEFAULT_THEME
    palette = THEMES[key]

    ACCENT = palette["accent"]
    ACCENT_DIM = palette["accent_dim"]
    INK = palette["ink"]
    SUBTLE = palette["subtle"]
    _active_theme = key

    # Mutated in place so existing THEME references stay valid
    THEME.update(
        {
            "brand": f"bold {ACCENT}",
            "accent": ACCENT,
            "user": SUBTLE,
            "user_border": SUBTLE,
            "user_title": SUBTLE,
            "assistant": INK,
            "assistant_border": ACCENT,
            "assistant_title": ACCENT,
            "system_border": SUBTLE,
            "tool": ACCENT,
            "tool_border": SUBTLE,
            "tool_title": ACCENT,
            "tool_result": SUBTLE,
            "error": palette["error"],
            "error_border": palette["error"],
            "error_title": f"bold {palette['error']}",
            "muted": "dim",
            "warn": palette["warn"],
            "ok": f"bold {palette['ok']}",
            "deny": f"bold {palette['error']}",
        }
    )
    return key


def active_theme() -> str:
    return _active_theme


apply_theme(DEFAULT_THEME)

TOOL_PREVIEW_CHARS = 400
TOOL_PREVIEW_LINES = 12
TOOL_ARGS_PREVIEW = 160
HISTORY_TOOL_CHARS = 200
HISTORY_TOOL_LINES = 4
FENCE_RE = re.compile(r"```(\w+)?\n", re.MULTILINE)


def _unicode_ok() -> bool:
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


# Gutter glyphs — ASCII fallback for legacy code pages
if _unicode_ok():
    GLYPH = {
        "bullet": "●",
        "branch": "⎿",
        "prompt": "❯",
        "star": "✻",
        "dot": "·",
        "arrow": "→",
    }
else:
    GLYPH = {
        "bullet": "*",
        "branch": "\\_",
        "prompt": ">",
        "star": "*",
        "dot": "-",
        "arrow": "->",
    }


# Descriptions for the command palette; unknown commands still list by name.
COMMAND_HINTS: dict[str, str] = {
    "/help": "Show help",
    "/clear": "Clear the screen",
    "/new": "Start a new conversation session",
    "/quiet": "Collapse tool output to one-liners",
    "/verbose": "Show full tool output",
    "/copy": "Copy last assistant reply",
    "/resume": "Resume a previous session",
    "/skills": "Manually pick skills (skips auto selection)",
    "/login": "Set Primary/Secondary API key",
    "/tavily": "Set Tavily API key for web_search",
    "/provider": "Change LLM provider",
    "/model": "Change model for active provider",
    "/history": "Set max messages kept in context",
    "/prices": "Set token price estimates",
    "/tokens": "Count tokens in session history",
    "/theme": "Change the color theme",
    "/exit": "End the session",
    "exit": "End the session",
    "quit": "End the session",
}

# Handled inside the console; never forwarded to the command router.
LOCAL_COMMANDS = ("/theme",)

PALETTE_SENTINEL = "\x00palette\x00"


def _theme_store_path() -> Optional[Path]:
    """Where the theme preference lives (inside the agent data root)."""
    try:
        from memory import get_data_root

        root = get_data_root()
        root.mkdir(parents=True, exist_ok=True)
        return root / "theme.txt"
    except Exception:
        return None


def _load_theme_preference() -> str:
    env_theme = (os.getenv("PI_THEME") or "").strip().lower()
    if env_theme in THEMES:
        return env_theme
    path = _theme_store_path()
    if path and path.is_file():
        try:
            saved = path.read_text(encoding="utf-8").strip().lower()
            if saved in THEMES:
                return saved
        except OSError:
            pass
    return DEFAULT_THEME


def _save_theme_preference(name: str) -> None:
    path = _theme_store_path()
    if not path:
        return
    try:
        path.write_text(name, encoding="utf-8")
    except OSError:
        pass


def _fuzzy_score(query: str, *fields: str) -> Optional[int]:
    """Subsequence match across fields. Lower score = better. None = no match."""
    if not query:
        return 0
    for rank, field in enumerate(fields):
        target = field.lower()
        if query in target:
            return rank * 100 + target.index(query)
        # subsequence fallback: /pv matches /provider
        pos, gaps, last = 0, 0, -1
        for ch in query:
            found = target.find(ch, pos)
            if found == -1:
                break
            if last >= 0:
                gaps += found - last - 1
            last = found
            pos = found + 1
        else:
            return rank * 100 + 50 + gaps
    return None


class TimedStatus:
    """Stable Rich Spinner with elapsed time (no custom frame/color thrashing)."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._start = time.time()
        # Fixed style — avoid dim/markup rebuilds that flicker on Windows consoles
        self._spinner = Spinner("dots", style=ACCENT)

    def __rich_console__(self, console, options):
        elapsed = int(time.time() - self._start)
        label = Text()
        label.append(f"{self.message} ", style=ACCENT)
        if "ESC" in self.message or "esc" in self.message:
            label.append(f"({elapsed}s)", style="dim")
        else:
            label.append(f"({elapsed}s {GLYPH['dot']} esc to interrupt)", style="dim")
        self._spinner.text = label
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
        self._at_gap: bool = True
        self.theme: str = apply_theme(_load_theme_preference())
        self._slash_commands: List[str] = [
            "/help",
            "/clear",
            "/quiet",
            "/verbose",
            "/copy",
            "/resume",
            "/login",
            "/tavily",
            "/provider",
            "/model",
            "/theme",
            "/exit",
            "exit",
            "quit",
        ]

    def set_slash_commands(self, commands: List[str]) -> None:
        """Update autocomplete list from Commands registry (plus console-local ones)."""
        names = [str(c) for c in commands]
        for local in LOCAL_COMMANDS:
            if local not in names:
                names.append(local)
        self._slash_commands = names

    # ------------------------------------------------------------------
    # Themes
    # ------------------------------------------------------------------

    def theme_names(self) -> List[str]:
        return list(THEMES)

    def set_theme(self, name: str, *, persist: bool = True, announce: bool = True) -> str:
        """Switch palette at runtime. Returns the theme actually applied."""
        applied = apply_theme(name)
        self.theme = applied
        if persist:
            _save_theme_preference(applied)
        if announce:
            self.print_theme_preview(applied)
        return applied

    def print_theme_preview(self, name: Optional[str] = None) -> None:
        """Small swatch so the user sees the palette immediately."""
        label = name or self.theme
        swatch = Text()
        swatch.append(f"{GLYPH['bullet']} accent  ", style=ACCENT)
        swatch.append("ink  ", style=INK)
        swatch.append("subtle  ", style=SUBTLE)
        swatch.append("warn  ", style=THEME["warn"])
        swatch.append("error", style=THEME["error"])

        header = Text()
        header.append("theme ", style="dim")
        header.append(label, style=f"bold {ACCENT}")

        self._emit(Group(header, swatch), gap_before=True)

    def _pick_theme(self) -> None:
        """Interactive theme chooser used by the console-local /theme command."""
        names = self.theme_names()
        labels = [
            f"{name}{'  (current)' if name == self.theme else ''}" for name in names
        ]
        picked = self.interactive_pick(
            labels,
            title="Select a theme",
            current=next((l for l in labels if "(current)" in l), None),
        )
        if not picked:
            self.print_system_message("Theme unchanged.", title="theme")
            return
        self.set_theme(picked.split("  (current)")[0].strip())

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
    # Layout helpers
    # ------------------------------------------------------------------

    def _emit(self, renderable: RenderableType, *, gap_before: bool = False) -> None:
        """Print a renderable, collapsing repeated blank separators."""
        if gap_before and not self._at_gap:
            self.console.print()
        self.console.print(renderable)
        self._at_gap = False

    def _gutter(
        self,
        marker: str,
        marker_style: str,
        body: RenderableType,
        indent: int = 0,
    ) -> Table:
        """Row of `marker  body` where wrapped body lines stay aligned."""
        marker_text = Text(" " * indent + marker, style=marker_style)
        grid = Table.grid(padding=(0, 1))
        grid.expand = True
        grid.add_column(width=len(marker_text.plain), no_wrap=True)
        grid.add_column(overflow="fold", ratio=1)
        grid.add_row(marker_text, body)
        return grid

    def _card(
        self, body: RenderableType, title: str = "", border: Optional[str] = None
    ) -> Panel:
        border = border or SUBTLE
        return Panel(
            body,
            title=f"[{border}]{title}[/{border}]" if title else None,
            title_align="left",
            border_style=border,
            box=box.ROUNDED,
            padding=(0, 1),
        )

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

    def _inline_args(self, arguments: str) -> str:
        """Compact `key: value` summary shown next to the tool name."""
        try:
            parsed = json.loads(arguments)
        except Exception:
            return (arguments or "").replace("\n", " ")[:60]

        if isinstance(parsed, dict):
            for key in ("path", "file", "filepath", "command", "query", "pattern", "url"):
                if key in parsed and isinstance(parsed[key], (str, int, float)):
                    return str(parsed[key]).replace("\n", " ")[:60]
            if parsed:
                first_key = next(iter(parsed))
                return f"{first_key}={str(parsed[first_key])}".replace("\n", " ")[:60]
        return str(parsed).replace("\n", " ")[:60]

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

    def print_help(self, command_rows: Optional[List[tuple[str, str]]] = None) -> None:
        """Print slash-command help. Rows come from Commands when available."""
        rows = command_rows or [
            ("/help", "Show this help"),
            ("/clear", "Clear the screen"),
            ("/quiet", "Collapse tool output to one-liners"),
            ("/verbose", "Show full tool output"),
            ("/copy", "Copy last assistant reply to clipboard"),
            ("/resume", "Resume a previous session"),
            ("/login", "Set Primary/Secondary API key (rate-limit failover)"),
            ("/tavily", "Set Tavily API key for web_search"),
            ("/provider", "Change LLM provider"),
            ("/model", "Change model for active provider"),
            ("/exit", "End the session"),
            ("exit / quit", "End the session"),
        ]

        listed = {cmd for cmd, _ in rows}
        for local in LOCAL_COMMANDS:
            if local not in listed:
                rows = list(rows) + [(local, COMMAND_HINTS.get(local, local))]

        commands = Table.grid(padding=(0, 2))
        commands.add_column(style=ACCENT, no_wrap=True)
        commands.add_column(style="dim", overflow="fold")
        for cmd, desc in rows:
            commands.add_row(cmd, desc)

        keys = Table.grid(padding=(0, 2))
        keys.add_column(style=INK, no_wrap=True)
        keys.add_column(style="dim", overflow="fold")
        for key, desc in [
            ("enter", "Submit task"),
            ("ctrl+p  or  /", "Open the command palette"),
            ("ctrl+j", "Insert newline"),
            ("esc esc+enter", "Insert newline (alt)"),
            ("esc", "Stop the current agent turn"),
            ("ctrl+c", "Exit"),
            ("up / down", "History and pickers"),
        ]:
            keys.add_row(key, desc)

        mode = "quiet" if self.quiet else ("verbose" if self.verbose else "normal")
        footer = Text()
        footer.append("tool output ", style="dim")
        footer.append(mode, style=ACCENT)
        footer.append("   theme ", style="dim")
        footer.append(self.theme, style=ACCENT)

        self._emit(
            Group(
                Text("Commands", style=f"bold {INK}"),
                commands,
                Text(""),
                Text("Keys", style=f"bold {INK}"),
                keys,
                Text(""),
                footer,
            ),
            gap_before=True,
        )
        self.console.print()
        self._at_gap = True

    def command_palette(self, initial_query: str = "") -> Optional[str]:
        """
        Fuzzy slash-command palette (type to filter, up/down, enter, esc).
        Returns the chosen command string, or None if cancelled.
        """
        self.stop_loading()
        entries = [
            (name, COMMAND_HINTS.get(name, ""))
            for name in self._slash_commands
            if name.startswith("/")
        ]
        if not entries:
            return None

        state = {"query": initial_query.lstrip("/"), "index": 0}

        def matches() -> List[tuple[str, str]]:
            query = state["query"].strip().lower()
            if not query:
                return entries
            scored = []
            for name, desc in entries:
                score = _fuzzy_score(query, name.lstrip("/"), desc)
                if score is not None:
                    scored.append((score, name, desc))
            scored.sort(key=lambda row: (row[0], row[1]))
            return [(name, desc) for _, name, desc in scored]

        def get_text() -> FormattedText:
            rows = matches()
            if rows:
                state["index"] %= len(rows)
            width = max((len(name) for name, _ in rows), default=10)

            fragments: list[tuple[str, str]] = [
                ("class:title", "Commands  "),
                ("class:muted", "type to filter  enter run  esc cancel\n"),
                ("class:prompt", f"{GLYPH['prompt']} "),
                ("class:query", f"{state['query']}\n\n"),
            ]
            if not rows:
                fragments.append(("class:muted", "  no matching command\n"))
            for i, (name, desc) in enumerate(rows[:12]):
                chosen = i == state["index"]
                marker = GLYPH["prompt"] if chosen else " "
                style = "class:selected" if chosen else "class:item"
                fragments.append((style, f"{marker} {name.ljust(width)}  "))
                fragments.append(
                    ("class:selected" if chosen else "class:muted", f"{desc}\n")
                )
            if len(rows) > 12:
                fragments.append(("class:muted", f"  ... {len(rows) - 12} more\n"))
            return FormattedText(fragments)

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            rows = matches()
            if rows:
                state["index"] = (state["index"] - 1) % len(rows)

        @kb.add("down")
        def _down(event) -> None:
            rows = matches()
            if rows:
                state["index"] = (state["index"] + 1) % len(rows)

        @kb.add("backspace")
        def _backspace(event) -> None:
            state["query"] = state["query"][:-1]
            state["index"] = 0

        @kb.add("c-u")
        def _clear(event) -> None:
            state["query"] = ""
            state["index"] = 0

        @kb.add("enter")
        def _enter(event) -> None:
            rows = matches()
            event.app.exit(result=rows[state["index"]][0] if rows else None)

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        @kb.add(Keys.Any)
        def _typed(event) -> None:
            char = event.data
            if char and char.isprintable():
                state["query"] += char
                state["index"] = 0

        style = PtStyle.from_dict(
            {
                "title": f"bold {ACCENT}",
                "prompt": f"bold {ACCENT}",
                "query": INK,
                "item": INK,
                "selected": f"bold {ACCENT}",
                "muted": f"italic {SUBTLE}",
            }
        )

        try:
            app: Application[Optional[str]] = Application(
                layout=Layout(
                    Window(FormattedTextControl(get_text), always_hide_cursor=True)
                ),
                key_bindings=kb,
                style=style,
                full_screen=False,
            )
            return app.run()
        except Exception:
            return None

    def _get_completer(self) -> Completer:
        commands = self._slash_commands

        class SlashCompleter(Completer):
            def get_completions(self, document, complete_event):
                word = document.get_word_before_cursor()
                text = document.text_before_cursor
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
        """Remove the leftover prompt line(s) so the user block isn't a duplicate."""
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

        @bindings.add("c-p")
        def _open_palette(event) -> None:
            # Nested apps can't run inside the prompt; exit and reopen after.
            event.app.exit(result=PALETTE_SENTINEL)

        pt_style = PtStyle.from_dict(
            {
                "prompt": f"{ACCENT} bold",
                "": INK,
            }
        )
        marker = FormattedText([("class:prompt", f"{GLYPH['prompt']} ")])

        while True:
            try:
                user_input = PtPrompt(
                    marker,
                    history=FileHistory(self.history_file),
                    auto_suggest=AutoSuggestFromHistory(),
                    completer=self._get_completer(),
                    complete_while_typing=True,
                    key_bindings=bindings,
                    style=pt_style,
                    multiline=False,
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.print_error(f"Advanced input failed: {str(e)}. Using basic input.")
                user_input = input(f"{GLYPH['prompt']} ")

            text = (user_input or "").strip()

            # ctrl+p, or a bare "/", opens the command palette
            if text == PALETTE_SENTINEL or text == "/":
                self._erase_prompt_echo(f"{GLYPH['prompt']} ")
                picked = self.command_palette("" if text == "/" else "")
                if not picked:
                    self._at_gap = True
                    continue
                text = picked
            elif text:
                # Drop the raw prompt echo; caller prints a single user line
                self._erase_prompt_echo(f"{GLYPH['prompt']} {user_input}")

            if not text:
                self._at_gap = True
                return ""

            # Console-local commands never reach the agent's command router
            if text.lower() in LOCAL_COMMANDS:
                self._handle_local_command(text.lower())
                self._at_gap = True
                continue

            self._at_gap = True
            return text

    def _handle_local_command(self, command: str) -> None:
        if command == "/theme":
            self._pick_theme()

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
            marker = GLYPH["prompt"] if i == selected["index"] else " "
            return f"{marker} {i + 1}. {title_text}  ({ws_short})"

        def get_text() -> FormattedText:
            fragments: list[tuple[str, str]] = [
                ("class:title", f"{title}\n"),
                ("class:muted", "up/down move  enter select  esc cancel  or type a number\n\n"),
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
            "title": f"bold {ACCENT}",
            "selected": f"bold {ACCENT}",
            "muted": f"italic {SUBTLE}",
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
        self._emit(Text(title, style=f"bold {ACCENT}"), gap_before=True)
        for i, item in enumerate(items, 1):
            self._emit(
                self._gutter(f"{i}.", ACCENT, Text(item.title, style=INK))
            )
        self.console.print()
        self._at_gap = True

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

    def interactive_pick(
        self,
        items: List[str],
        title: str = "Select an option",
        current: Optional[str] = None,
        window_size: int = 12,
        *,
        searchable: bool = False,
        custom_option: Optional[str] = None,
    ) -> Optional[str]:
        """
        Arrow-key picker for string options.
        Returns the selected string, or None if cancelled.

        searchable: type to fuzzy-filter the list.
        custom_option: always-available extra row (e.g. "+ Enter model name...").
        """
        self.stop_loading()
        base_items = list(items)
        if custom_option and custom_option not in base_items:
            base_items = base_items + [custom_option]
        if not base_items:
            raise ValueError("No items to select from")

        state = {
            "index": 0,
            "query": "",
        }
        if current and current in base_items:
            state["index"] = base_items.index(current)

        def filtered() -> List[str]:
            query = state["query"].strip().lower()
            if not searchable or not query:
                return list(base_items)
            scored: list[tuple[int, str]] = []
            for label in base_items:
                if custom_option and label == custom_option:
                    # Keep custom option visible while searching
                    scored.append((10_000, label))
                    continue
                score = _fuzzy_score(query, label)
                if score is not None:
                    scored.append((score, label))
            scored.sort(key=lambda row: (row[0], row[1].lower()))
            return [label for _, label in scored]

        def _visible_slice(rows: List[str]) -> tuple[int, int]:
            n = len(rows)
            if n <= window_size:
                return 0, n
            half = window_size // 2
            start = max(0, state["index"] - half)
            end = min(n, start + window_size)
            start = max(0, end - window_size)
            return start, end

        def get_text() -> FormattedText:
            rows = filtered()
            if rows:
                state["index"] %= len(rows)
            else:
                state["index"] = 0

            hints = "up/down move  enter select  esc cancel"
            if searchable:
                hints = "type to search  " + hints
            fragments: list[tuple[str, str]] = [
                ("class:title", f"{title}\n"),
                ("class:muted", f"{hints}\n"),
            ]
            if searchable:
                fragments.append(("class:prompt", f"{GLYPH['prompt']} "))
                fragments.append(("class:query", f"{state['query']}\n"))
            fragments.append(("", "\n"))

            if not rows:
                fragments.append(("class:muted", "  no matches — try another search\n"))
                return FormattedText(fragments)

            start, end = _visible_slice(rows)
            if start > 0:
                fragments.append(("class:muted", f"  ... {start} more above\n"))
            for i in range(start, end):
                marker = GLYPH["prompt"] if i == state["index"] else " "
                label = rows[i]
                suffix = "  (current)" if current and label == current else ""
                style = "class:selected" if i == state["index"] else ""
                fragments.append((style, f"{marker} {label}{suffix}\n"))
            if end < len(rows):
                fragments.append(("class:muted", f"  ... {len(rows) - end} more below\n"))
            return FormattedText(fragments)

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            rows = filtered()
            if rows:
                state["index"] = (state["index"] - 1) % len(rows)

        @kb.add("down")
        def _down(event) -> None:
            rows = filtered()
            if rows:
                state["index"] = (state["index"] + 1) % len(rows)

        @kb.add("enter")
        def _enter(event) -> None:
            rows = filtered()
            event.app.exit(result=rows[state["index"]] if rows else None)

        @kb.add("escape")
        def _esc(event) -> None:
            event.app.exit(result=None)

        if searchable:
            @kb.add("backspace")
            def _backspace(event) -> None:
                state["query"] = state["query"][:-1]
                state["index"] = 0

            @kb.add("c-u")
            def _clear(event) -> None:
                state["query"] = ""
                state["index"] = 0

            @kb.add(Keys.Any)
            def _typed(event) -> None:
                char = event.data
                if char and char.isprintable():
                    state["query"] += char
                    state["index"] = 0

        style = PtStyle.from_dict({
            "title": f"bold {ACCENT}",
            "selected": f"bold {ACCENT}",
            "muted": f"italic {SUBTLE}",
            "prompt": f"bold {ACCENT}",
            "query": INK,
        })

        try:
            app: Application[Optional[str]] = Application(
                layout=Layout(Window(FormattedTextControl(get_text), always_hide_cursor=True)),
                key_bindings=kb,
                style=style,
                full_screen=False,
            )
            return app.run()
        except Exception:
            # Numeric / text fallback
            self._emit(Text(title, style=f"bold {ACCENT}"), gap_before=True)
            for i, item in enumerate(base_items, 1):
                mark = "  (current)" if current and item == current else ""
                self._emit(
                    self._gutter(f"{i}.", ACCENT, Text(f"{item}{mark}", style=INK))
                )
            while True:
                try:
                    from rich.prompt import Prompt

                    hint = "number"
                    if searchable:
                        hint = "number or search text"
                    if custom_option:
                        hint += f", or '{len(base_items)}' for custom"
                    choice = Prompt.ask(
                        f"Enter {hint} (blank to cancel)",
                        console=self.console,
                        default="",
                    )
                    if not choice.strip():
                        return None
                    if choice.strip().isdigit():
                        idx = int(choice.strip()) - 1
                        if 0 <= idx < len(base_items):
                            return base_items[idx]
                        self.print_error(f"Choose 1-{len(base_items)}.")
                        continue
                    # Treat as search: pick best fuzzy match, or custom if requested
                    query = choice.strip().lower()
                    if custom_option and query in ("custom", "manual", "other"):
                        return custom_option
                    scored = []
                    for label in base_items:
                        if custom_option and label == custom_option:
                            continue
                        score = _fuzzy_score(query, label)
                        if score is not None:
                            scored.append((score, label))
                    if scored:
                        scored.sort(key=lambda row: (row[0], row[1].lower()))
                        return scored[0][1]
                    self.print_error("No match. Enter a number or clearer search text.")
                except ValueError:
                    self.print_error("Please enter a valid number.")

    def interactive_multi_pick(
        self,
        items: List[str],
        title: str = "Select options",
        preselected: Optional[List[str]] = None,
        window_size: int = 14,
    ) -> Optional[tuple[str, List[str]]]:
        """
        Multi-select picker.

        Returns:
          None — cancelled
          ("auto", []) — user chose auto mode
          ("manual", [...]) — confirmed selection (may be empty = no skills)
        """
        self.stop_loading()
        if not items:
            raise ValueError("No items to select from")

        selected = {"index": 0}
        checked: set[str] = set(preselected or []) & set(items)
        outcome: dict[str, Any] = {"mode": "manual"}

        def _visible_slice() -> tuple[int, int]:
            n = len(items)
            if n <= window_size:
                return 0, n
            half = window_size // 2
            start = max(0, selected["index"] - half)
            end = min(n, start + window_size)
            start = max(0, end - window_size)
            return start, end

        def get_text() -> FormattedText:
            fragments: list[tuple[str, str]] = [
                ("class:title", f"{title}\n"),
                (
                    "class:muted",
                    "up/down move  space toggle  enter confirm  "
                    "c clear all  a auto mode  esc cancel\n"
                    f"selected: {len(checked)}\n\n",
                ),
            ]
            start, end = _visible_slice()
            if start > 0:
                fragments.append(("class:muted", f"  ... {start} more above\n"))
            for i in range(start, end):
                name = items[i]
                cursor = GLYPH["prompt"] if i == selected["index"] else " "
                mark = "[x]" if name in checked else "[ ]"
                style = "class:selected" if i == selected["index"] else ""
                fragments.append((style, f"{cursor} {mark} {name}\n"))
            if end < len(items):
                fragments.append(("class:muted", f"  ... {len(items) - end} more below\n"))
            return FormattedText(fragments)

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            selected["index"] = (selected["index"] - 1) % len(items)

        @kb.add("down")
        def _down(event) -> None:
            selected["index"] = (selected["index"] + 1) % len(items)

        @kb.add("space")
        def _toggle(event) -> None:
            name = items[selected["index"]]
            if name in checked:
                checked.discard(name)
            else:
                checked.add(name)

        @kb.add("c")
        @kb.add("C")
        def _clear(event) -> None:
            checked.clear()

        @kb.add("a")
        @kb.add("A")
        def _auto(event) -> None:
            outcome["mode"] = "auto"
            event.app.exit(result=[])

        @kb.add("enter")
        def _enter(event) -> None:
            outcome["mode"] = "manual"
            event.app.exit(result=[name for name in items if name in checked])

        @kb.add("escape")
        def _esc(event) -> None:
            event.app.exit(result=None)

        style = PtStyle.from_dict({
            "title": f"bold {ACCENT}",
            "selected": f"bold {ACCENT}",
            "muted": f"italic {SUBTLE}",
        })

        try:
            app: Application[Optional[List[str]]] = Application(
                layout=Layout(Window(FormattedTextControl(get_text), always_hide_cursor=True)),
                key_bindings=kb,
                style=style,
                full_screen=False,
            )
            picked = app.run()
            if picked is None:
                return None
            if outcome["mode"] == "auto":
                return ("auto", [])
            return ("manual", picked)
        except Exception:
            self._emit(Text(title, style=f"bold {ACCENT}"), gap_before=True)
            for i, item in enumerate(items, 1):
                mark = "x" if item in checked else " "
                self._emit(
                    self._gutter(f"{i}.", ACCENT, Text(f"[{mark}] {item}", style=INK))
                )
            self._emit(
                Text(
                    "Enter numbers (e.g. 1,3,5), 'a' for auto, or blank to cancel",
                    style="dim",
                )
            )
            while True:
                try:
                    from rich.prompt import Prompt

                    choice = Prompt.ask("Skills", console=self.console, default="").strip()
                    if not choice:
                        return None
                    if choice.lower() == "a":
                        return ("auto", [])
                    indices = []
                    for part in choice.replace(",", " ").split():
                        indices.append(int(part) - 1)
                    picked_names: List[str] = []
                    for idx in indices:
                        if not (0 <= idx < len(items)):
                            raise ValueError(idx)
                        if items[idx] not in picked_names:
                            picked_names.append(items[idx])
                    return ("manual", picked_names)
                except ValueError:
                    self.print_error(f"Enter valid numbers between 1 and {len(items)}.")

    def prompt_text(self, label: str, default: str = "") -> Optional[str]:
        """Prompt for a single line of text. Returns None on cancel/empty when no default."""
        self.stop_loading()
        marker = FormattedText(
            [
                ("class:label", f"{label} "),
                ("class:prompt", f"{GLYPH['prompt']} "),
            ]
        )
        pt_style = PtStyle.from_dict({"label": SUBTLE, "prompt": f"{ACCENT} bold", "": INK})
        try:
            value = PtPrompt(marker, default=default, style=pt_style).strip()
        except KeyboardInterrupt:
            return None
        except Exception:
            from rich.prompt import Prompt

            value = Prompt.ask(label, default=default, console=self.console).strip()
        self._at_gap = False
        return value or None

    # ------------------------------------------------------------------
    # Message printers
    # ------------------------------------------------------------------

    def print_system_message(self, message: str, title: str = "System"):
        text = message or ""
        multiline = "\n" in text.strip()

        if not multiline:
            body = Text()
            if title and title.lower() not in ("system", ""):
                body.append(f"{title}  ", style=ACCENT)
            body.append(text.strip(), style="dim")
            self._emit(self._gutter(GLYPH["dot"], SUBTLE, body), gap_before=True)
            return

        self._emit(
            self._card(Text(text, style=INK), title=title, border=SUBTLE),
            gap_before=True,
        )

    def print_user_message(self, message: str):
        self._emit(
            self._gutter(GLYPH["prompt"], ACCENT, Text(message, style=SUBTLE)),
            gap_before=True,
        )

    def print_assistant_message(self, message: str, is_code: bool = False):
        """Print assistant response; skips empty content; detects fenced languages."""
        content = message or ""
        if content.strip():
            self._last_assistant_message = content

        if not content.strip():
            self._emit(
                self._gutter(GLYPH["bullet"], "dim", Text("no text returned", style="dim")),
                gap_before=True,
            )
            return

        if is_code:
            lang = self._detect_code_language(content) or "python"
            try:
                body: RenderableType = Syntax(content, lang, theme="ansi_dark", line_numbers=False)
                self._emit(self._gutter(GLYPH["bullet"], ACCENT, body), gap_before=True)
                return
            except Exception:
                pass

        try:
            body = Markdown(content)
        except Exception:
            body = Text(content, style=INK)

        self._emit(self._gutter(GLYPH["bullet"], ACCENT, body), gap_before=True)

    def print_tool_call(self, tool_name: str, arguments: str):
        with self._pause_loading():
            summary = self._inline_args(arguments)
            header = Text()
            header.append(tool_name, style=f"bold {INK}")
            if summary:
                header.append(f"({summary})", style="dim")

            self._emit(self._gutter(GLYPH["bullet"], ACCENT, header), gap_before=True)

            # The header already carries the key argument; full args are opt-in
            if not self.verbose or not arguments:
                return

            args_display, _ = self._pretty_args(arguments)
            if not args_display.strip():
                return

            self._emit(
                self._gutter(
                    GLYPH["branch"], SUBTLE, Text(args_display, style="dim"), indent=1
                )
            )

    def print_tool_result(self, result: str):
        with self._pause_loading():
            text = result or ""
            if self.quiet:
                preview, _ = self._truncate(text.replace("\n", " "), 100, 1)
                self._emit(
                    self._gutter(GLYPH["branch"], SUBTLE, Text(preview, style="dim"), indent=1)
                )
                return

            if self.verbose:
                body, truncated = text, False
            else:
                body, truncated = self._truncate(text, TOOL_PREVIEW_CHARS, TOOL_PREVIEW_LINES)

            if truncated:
                body = body.rstrip().removesuffix("...").rstrip()

            renderable: RenderableType = Text(body, style=THEME["tool_result"])
            if truncated:
                hidden = max(0, len(text.splitlines()) - TOOL_PREVIEW_LINES)
                note = (
                    f"+{hidden} lines (/verbose to expand)"
                    if hidden
                    else "truncated (/verbose to expand)"
                )
                renderable = Group(renderable, Text(note, style="dim"))

            self._emit(self._gutter(GLYPH["branch"], SUBTLE, renderable, indent=1))

    def print_error(self, error: str, title: str = "Error"):
        with self._pause_loading():
            body = Text()
            if title and title.lower() != "error":
                body.append(f"{title}: ", style="bold red")
            body.append(error, style="red")
            self._emit(self._gutter(GLYPH["bullet"], "red", body), gap_before=True)

    def print_welcome(
        self,
        title: str = " ",
        ws_path: str = "",
        is_auth: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ):
        if title and title.strip():
            self._session_title = title
        if ws_path:
            self._session_workspace = ws_path

        header = Text()
        header.append(f"{GLYPH['star']} ", style=ACCENT)
        header.append("PI", style=f"bold {ACCENT}")
        header.append("  Python Agent Harness", style="dim")

        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="dim", no_wrap=True)
        meta.add_column(style=INK, overflow="fold")
        if self._session_title.strip():
            meta.add_row("session", self._session_title.strip())
        if self._session_workspace:
            meta.add_row("cwd", self._session_workspace)
        if provider or model:
            meta.add_row("model", f"{provider or '-'} {GLYPH['dot']} {model or '-'}")

        blocks: list[RenderableType] = [header]
        if meta.row_count:
            blocks.extend([Text(""), meta])

        self.console.print()
        self.console.print(self._card(Group(*blocks), border=ACCENT_DIM))

        hints = Text()
        hints.append("  /help", style=ACCENT)
        hints.append(" for commands", style="dim")
        hints.append(f"   {GLYPH['dot']} esc to interrupt", style="dim")
        hints.append(f"   {GLYPH['dot']} ctrl+j for newline", style="dim")
        self.console.print(hints)
        self.console.print()
        self._at_gap = True

        if not is_auth:
            warning = Text()
            warning.append("not authenticated", style="bold yellow")
            warning.append("  run ", style="dim")
            warning.append("/login", style=ACCENT)
            warning.append(" to add a provider API key", style="dim")
            self._emit(self._gutter(GLYPH["bullet"], "yellow", warning))
            self.console.print()
            self._at_gap = True

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def confirm_permission(self, action_details: str, title: str = "Permission Required") -> bool:
        was_loading = self._current_live is not None
        if was_loading:
            self.stop_loading()

        self._emit(
            self._card(Text(action_details, style=INK), title=title, border="yellow"),
            gap_before=True,
        )
        from rich.prompt import Confirm

        res = Confirm.ask("Allow this action?", console=self.console, default=False)
        self._at_gap = False

        if was_loading:
            self.start_loading("Working")
        return res

    def confirm_permission_extended(self, tool_name: str, target: str, action_details: str) -> str:
        """
        Single-key permission prompt.
        Returns: 'y' | 'a' | 'f' | 'all' | 'n'  (same contract as before).
        Keys: y, a, f, l (all), n — or 1-5. Enter not required.
        """
        was_loading = self._current_live is not None
        if was_loading:
            self.stop_loading()

        clean_target = target[:60] + "..." if len(target) > 60 else target
        risky = tool_name in ("bash", "write", "edit")
        border = "red" if risky else "yellow"

        options = [
            ("1", "y", "Yes, once"),
            ("2", "a", f"Yes, always for {tool_name}"),
            ("3", "f", f"Yes, always for {clean_target}"),
            ("4", "l", "Yes, always for every tool"),
            ("5", "n", "No, deny"),
        ]

        choices = Table.grid(padding=(0, 2))
        choices.add_column(style=ACCENT, no_wrap=True)
        choices.add_column(style=INK, overflow="fold")
        for number, key, label in options:
            choices.add_row(f"{number} / {key}", label)

        body = Group(
            Text(action_details, style=INK),
            Text(""),
            choices,
        )
        self._emit(
            self._card(body, title=f"permission {GLYPH['dot']} {tool_name}", border=border),
            gap_before=True,
        )

        key_map = {
            "y": "y",
            "a": "a",
            "f": "f",
            "l": "all",
            "n": "n",
            "1": "y",
            "2": "a",
            "3": "f",
            "4": "all",
            "5": "n",
        }
        result = {"value": "n"}

        kb = KeyBindings()

        for key, value in key_map.items():
            @kb.add(key)
            def _pick(event, value=value) -> None:
                result["value"] = value
                event.app.exit()

            if key.isalpha():
                @kb.add(key.upper())
                def _pick_upper(event, value=value) -> None:
                    result["value"] = value
                    event.app.exit()

        @kb.add("escape")
        def _deny(event) -> None:
            result["value"] = "n"
            event.app.exit()

        try:
            self.console.print(Text(f"{GLYPH['prompt']} ", style=f"bold {ACCENT}"), end="")
            app: Application[None] = Application(
                layout=Layout(Window(FormattedTextControl(lambda: ""), height=1)),
                key_bindings=kb,
                full_screen=False,
            )
            app.run()
            self.console.print(Text(result["value"], style=INK))
            self._at_gap = False
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
            self.start_loading("Working")

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
        line = Text()
        line.append(f"{total_tokens:,} tokens", style="dim")
        line.append(f"  in {prompt_tokens:,} {GLYPH['dot']} out {completion_tokens:,}", style="dim")
        line.append(f"  ~${estimated_cost_usd:.4f}", style="dim")
        self._emit(self._gutter(" ", "dim", line))

    def print_separator(self):
        """Breathing room between turns (collapses when already spaced)."""
        if not self._at_gap:
            self.console.print()
            self._at_gap = True

    def clear_screen(self):
        self.console.clear()
        self._at_gap = True

    def print_code_block(self, code: str, language: str = "python"):
        try:
            self._emit(
                Syntax(code, language, theme="ansi_dark", line_numbers=False),
                gap_before=True,
            )
        except Exception:
            self._emit(Text(code, style=INK), gap_before=True)

    def print_chat_history(self, messages: List[Any]) -> None:
        """Replay history with system prompt collapsed and tool results trimmed."""
        if not messages:
            self._emit(
                self._gutter(GLYPH["dot"], SUBTLE, Text("no chat history found", style="dim")),
                gap_before=True,
            )
            return

        self._emit(Text("history", style="dim"), gap_before=True)

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
                    self._emit(
                        self._gutter(
                            GLYPH["bullet"], "dim", Text(f"{n} tool call(s)", style="dim")
                        ),
                        gap_before=True,
                    )
                else:
                    self.print_assistant_message(content)
            elif role == "system":
                preview, _ = self._truncate(content.replace("\n", " "), 100, 1)
                body = Text()
                body.append("system prompt  ", style="dim")
                body.append(preview, style="dim")
                self._emit(self._gutter(GLYPH["dot"], SUBTLE, body), gap_before=True)
            elif role == "tool":
                name = getattr(msg, "name", None) or "tool"
                body_text, truncated = self._truncate(
                    content, HISTORY_TOOL_CHARS, HISTORY_TOOL_LINES
                )
                if truncated:
                    body_text = body_text.rstrip(".") + "\n... (collapsed)"
                detail = Text()
                detail.append(f"{name}  ", style="dim")
                detail.append(body_text, style=THEME["tool_result"])
                self._emit(self._gutter(GLYPH["branch"], SUBTLE, detail, indent=1))

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
        hint = f" (or set {env_var_hint} in environment)" if env_var_hint else ""

        info = Text()
        info.append(f"{provider} API key", style=f"bold {INK}")
        info.append(f"  input is hidden{hint}", style="dim")
        self._emit(self._gutter(GLYPH["bullet"], ACCENT, info), gap_before=True)

        pt_style = PtStyle.from_dict({"prompt": f"{ACCENT} bold", "": INK})
        marker = FormattedText([("class:prompt", f"{GLYPH['prompt']} ")])

        while True:
            try:
                api_key = PtPrompt(
                    marker,
                    is_password=True,
                    style=pt_style,
                ).strip()
            except KeyboardInterrupt:
                raise
            except Exception:
                from rich.prompt import Prompt

                api_key = Prompt.ask(
                    f"{provider} API Key",
                    password=True,
                    console=self.console,
                ).strip()

            if api_key:
                self.print_system_message(f"{provider} key saved.", title="auth")
                return api_key

            self.print_error("API key cannot be empty.", title="auth")


# Global console instance
console_ui = ConsoleUI()


def get_console() -> ConsoleUI:
    """Get the global console instance."""
    return console_ui
