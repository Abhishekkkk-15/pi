import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any


def _kill_process_tree(pid: int):
    """Terminates a process tree cleanly cross-platform."""
    try:
        if sys.platform == "win32":
            subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


def execute_read(path: str) -> str:
    """Reads and returns the contents of a text file."""
    try:
        filepath = Path(path)
        if not filepath.exists():
            return f"Error: File '{path}' does not exist."
        if not filepath.is_file():
            return f"Error: '{path}' is a directory, not a file."
        
        return filepath.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


def execute_write(path: str, content: str) -> str:
    """Creates or completely overwrites a file with the given content."""
    try:
        filepath = Path(path)
        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        return f"Successfully wrote to '{path}'."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"


def execute_edit(path: str, edits: List[Dict[str, str]]) -> str:
    """
    Applies exact search-and-replace edits to a file.
    Note: All edits operate on the original file content simultaneously.
    """
    try:
        filepath = Path(path)
        if not filepath.exists():
            return f"Error: File '{path}' does not exist."
        
        content = filepath.read_text(encoding="utf-8")

        for i, edit in enumerate(edits):
            old_text = edit.get("oldText", "")
            new_text = edit.get("newText", "")

            if old_text not in content:
                return (
                    f"Error in edit entry {i + 1}: Could not find exact match for 'oldText'.\n"
                    f"Target text was:\n{old_text}"
                )
            
            # Check for multiple occurrences
            occurrences = content.count(old_text)
            if occurrences > 1:
                return (
                    f"Error in edit entry {i + 1}: 'oldText' matched {occurrences} locations. "
                    "Provide more surrounding context in 'oldText' to make it unique."
                )

            # Replace the exact block
            content = content.replace(old_text, new_text, 1)

        filepath.write_text(content, encoding="utf-8")
        return f"Successfully applied {len(edits)} edit(s) to '{path}'."

    except Exception as e:
        return f"Error editing file '{path}': {str(e)}"


def execute_bash(command: str, timeout: int = 30, is_background: bool = False) -> str:
    """
    Executes a terminal command cross-platform without hanging or freezing.
    Auto-detects or handles long-running server commands (e.g. npm run dev, vite),
    prevents interactive CLI prompt hangs using stdin=DEVNULL,
    and kills processes cleanly on timeout to prevent lingering process leaks.
    """
    bg_keywords = [
        "npm run dev", "npm start", "vite", "next dev", "ng serve",
        "gatsby develop", "nodemon", "uvicorn", "gunicorn", "flask run",
        "python -m http.server"
    ]
    command_lower = command.lower()
    auto_bg = any(kw in command_lower for kw in bg_keywords)
    should_run_bg = is_background or auto_bg

    try:
        env = os.environ.copy()
        env["CI"] = "true"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["npm_config_yes"] = "true"
        env["NONINTERACTIVE"] = "1"
        env["FORCE_COLOR"] = "0"
        env["NO_COLOR"] = "1"
        env["PIP_NO_INPUT"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"

        creationflags = 0
        if sys.platform == "win32" and should_run_bg:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,  # Prevents interactive CLI prompts from blocking on stdin
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []

        def _read_stream(stream, output_list):
            try:
                for line in iter(stream.readline, ''):
                    output_list.append(line)
                stream.close()
            except Exception:
                pass

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines), daemon=True)
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines), daemon=True)
        t_out.start()
        t_err.start()

        wait_limit = 4 if should_run_bg else timeout
        start_time = time.time()

        while time.time() - start_time < wait_limit:
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        is_running = proc.poll() is None

        if is_running:
            if should_run_bg:
                output = "".join(stdout_lines) + "".join(stderr_lines)
                output_str = output.strip() if output else "[Process started successfully]"
                return (
                    f"{output_str}\n\n"
                    f"[Background process started and running with PID {proc.pid}]"
                )
            else:
                output = "".join(stdout_lines) + "".join(stderr_lines)
                output_str = output.strip() if output else "[No output received before timeout]"
                _kill_process_tree(proc.pid)
                return (
                    f"{output_str}\n\n"
                    f"[Error: Command timed out after {timeout} seconds and was terminated.]"
                )

        output = "".join(stdout_lines) + "".join(stderr_lines)
        return output.strip() if output else "[Command finished with no output]"

    except Exception as e:
        return f"Error executing command: {str(e)}"


def execute_web_search(query: str, max_results: int = 5) -> str:
    """Executes a real-time web search using the Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable is not set. Please add TAVILY_API_KEY to your .env file."

    try:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=max_results)
            results = response.get("results", [])
        except ImportError:
            import urllib.request
            import json
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=json.dumps({"api_key": api_key, "query": query, "max_results": max_results}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])

        if not results:
            return f"No web search results found for query: '{query}'."

        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            url = res.get("url", "")
            content = res.get("content", "")
            formatted_results.append(f"[{i}] {title}\nURL: {url}\nContent: {content}\n")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error executing web search: {str(e)}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute file path to read."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create a new file or completely overwrite an existing file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write to."
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete text content to write into the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Make precise, surgical changes to a file by providing exact original text blocks to replace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit."
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text block from the file to be replaced."
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "New text to replace oldText with."
                                }
                            },
                            "required": ["oldText", "newText"]
                        },
                        "description": "List of precise edits to perform."
                    }
                },
                "required": ["path", "edits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run shell/bash commands (ls, git, pytest, grep, find, npm, etc.). Non-blocking for background/dev server commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command string to execute."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum time in seconds to wait for command completion (default: 30)."
                    },
                    "is_background": {
                        "type": "boolean",
                        "description": "Set to true for long-running background tasks or dev servers (e.g., npm run dev)."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform real-time web searches using Tavily for up-to-date documentation, news, or answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default: 5)."
                    }
                },
                "required": ["query"]
            }
        }
    }
]