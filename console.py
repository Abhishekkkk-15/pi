"""
Enhanced Console UI for pi-python agent
Provides rich formatting, command history, and better UX
"""
from contextlib import contextmanager   # add at top

import sys
import os
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich import box
from prompt_toolkit import prompt as PtPrompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from pathlib import Path


class ConsoleUI:
    """Enhanced console interface for the pi agent"""
    
    def __init__(self, history_file: str = ".pi_history"):
        self.console = Console()
        self.history_file = history_file
        self._setup_history()
        
    def _setup_history(self):
        """Setup command history file"""
        history_path = Path(self.history_file)
        if not history_path.exists():
            history_path.touch()
    
    def print_system_message(self, message: str, title: str = "System"):
        """Print a system message in a styled panel"""
        self.console.print(
            Panel(
                Text(message, style="white"),
                title=title,
                border_style="blue",
                box=box.DOUBLE
            )
        )
    
    def print_user_message(self, message: str):
        """Print user message with styling"""
        self.console.print(
            Panel(
                Text(message, style="cyan"),
                title="[bold green]User[/bold green]",
                border_style="green",
                box=box.ROUNDED
            )
        )
    
    def print_assistant_message(self, message: str, is_code: bool = False):
        """Print assistant response with proper formatting"""
        if is_code:
            # Try to detect language and apply syntax highlighting
            try:
                syntax = Syntax(message, "python", theme="monokai", line_numbers=True)
                self.console.print(
                    Panel(
                        syntax,
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )
            except:
                # Fallback to regular text
                self.console.print(
                    Panel(
                        Text(message, style="white"),
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )
        else:
            # Try to render as markdown
            try:
                md = Markdown(message)
                self.console.print(
                    Panel(
                        md,
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )
            except:
                self.console.print(
                    Panel(
                        Text(message, style="white"),
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        box=box.ROUNDED
                    )
                )
    
    def print_tool_call(self, tool_name: str, arguments: str):
        """Print tool call information"""
        self.console.print(
            f"[bold yellow]>>[/bold yellow] Calling tool: [cyan]{tool_name}[/cyan] with args: [white]{arguments}[/white]"
        )
    
    def print_tool_result(self, result: str):
        """Print tool execution result"""
        self.console.print(
            Panel(
                Text(result, style="light_green"),
                title="[bold yellow]Tool Result[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED
            )
        )
    
    def print_error(self, error: str, title: str = "Error"):
        """Print error message"""
        self.console.print(
            Panel(
                Text(error, style="red"),
                title=f"[bold red]{title}[/bold red]",
                border_style="red",
                box=box.HEAVY
            )
        )
    
    def print_welcome(self):
        """Print welcome message"""
        welcome_text = Text()
        welcome_text.append("\n")
        welcome_text.append("  PI - Python Agent Harness  ", style="bold white on blue")
        welcome_text.append("\n")
        welcome_text.append("  Enhanced Console Interface  ", style="bold cyan")
        welcome_text.append("\n")
        welcome_text.append("  Type 'exit' or 'quit' to end session  ", style="dim")
        welcome_text.append("\n")
        
        self.console.print(Panel(welcome_text, box=box.DOUBLE))
        
    @contextmanager
    def print_loading(self, message: str = "Thinking..."):
        """Show loading spinner"""
        with Live(Spinner("dots", text=message), console=self.console) as live:
            yield live
    
    def get_user_input(self, prompt: str = "Enter your task") -> str:
        """Get user input with history and auto-suggest"""
        try:
            # Use prompt_toolkit for better input experience
            user_input = PtPrompt(
                f"{prompt} > ",
                history=FileHistory(self.history_file),
                auto_suggest=AutoSuggestFromHistory(),
                completer=self._get_completer(),
                complete_while_typing=True
            )
            return user_input.strip()
        except Exception as e:
            # Fallback to regular input
            self.print_error(f"Advanced input failed: {str(e)}. Using basic input.")
            return input(f"{prompt} > ").strip()
    
    def _get_completer(self) -> Completer:
        """Get completer for common commands"""
        class SimpleCompleter(Completer):
            def get_completions(self, document, complete_event) -> List[Completion]:
                commands = ["read", "write", "edit", "bash", "exit", "quit", "help", "clear"]
                word = document.get_word_before_cursor()
                for cmd in commands:
                    if cmd.startswith(word):
                        yield Completion(cmd, start_position=-len(word))
        
        return SimpleCompleter()
    
    def print_separator(self):
        """Print a separator line"""
        self.console.print("─" * self.console.width, style="dim")
    
    def clear_screen(self):
        """Clear the console screen"""
        self.console.clear()
        self.print_welcome()
    
    def print_code_block(self, code: str, language: str = "python"):
        """Print a code block with syntax highlighting"""
        try:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        except:
            self.console.print(f"[dim]Code block ({language}):[/dim]")
            self.console.print(code)


# Global console instance
console_ui = ConsoleUI()


def get_console() -> ConsoleUI:
    """Get the global console instance"""
    return console_ui
