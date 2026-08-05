import os
import re
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any


SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".pi-python",
    ".memory",
    ".cursor",
}

SKIP_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


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
        existed = filepath.exists() and filepath.is_file()
        # Create parent directories if they don't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        text = content or ""
        filepath.write_text(text, encoding="utf-8")
        lines = len(text.splitlines()) if text else 0
        nbytes = len(text.encode("utf-8"))
        action = "Overwrote" if existed else "Created"
        return f"{action} '{path}' - {lines} lines, {nbytes} bytes."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"


def execute_edit(path: str, edits: List[Dict[str, str]]) -> str:
    """
    Applies parallel exact search-and-replace edits to a file.
    All edits are matched against the original file state before any modifications occur.
    """
    try:
        filepath = Path(path)
        if not filepath.exists():
            return f"Error: File '{path}' does not exist."

        content = filepath.read_text(encoding="utf-8")
        # 1. Normalize CRLF to LF for matching consistency
        normalized_content = content.replace("\r\n", "\n")
        
        matches = []
        for i, edit in enumerate(edits):
            old_text = (edit.get("oldText", "") or "").replace("\r\n", "\n")
            new_text = (edit.get("newText", "") or "").replace("\r\n", "\n")

            if not old_text:
                return f"Error in edit {i + 1}: 'oldText' cannot be empty."

            # Check uniqueness in original content
            occurrences = normalized_content.count(old_text)
            if occurrences == 0:
                return (
                    f"Error in edit entry {i + 1}: Could not find exact match for 'oldText'.\n"
                    f"Target text was:\n{old_text}"
                )
            if occurrences > 1:
                return (
                    f"Error in edit entry {i + 1}: 'oldText' matched {occurrences} locations. "
                    "Provide more surrounding context to make it unique."
                )

            start_idx = normalized_content.index(old_text)
            end_idx = start_idx + len(old_text)
            matches.append({
                "index": i,
                "start": start_idx,
                "end": end_idx,
                "new_text": new_text,
                "old_text": old_text,
            })

        # 2. Check for overlapping edit regions
        matches.sort(key=lambda m: m["start"])
        for i in range(1, len(matches)):
            prev = matches[i - 1]
            curr = matches[i]
            if prev["end"] > curr["start"]:
                return (
                    f"Error: Edits {prev['index'] + 1} and {curr['index'] + 1} overlap in '{path}'. "
                    "Merge them into a single replacement block."
                )

        # 3. Apply edits in reverse order (highest start index down to 0)
        new_content = normalized_content
        for m in reversed(matches):
            new_content = new_content[:m["start"]] + m["new_text"] + new_content[m["end"]:]

        # Restore CRLF line endings if original file used CRLF
        if "\r\n" in content:
            new_content = new_content.replace("\n", "\r\n")

        filepath.write_text(new_content, encoding="utf-8")
        return f"Applied {len(edits)} edit(s) to '{path}' successfully."

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
    from config import get_tavily_api_key

    api_key = get_tavily_api_key()
    if not api_key:
        return (
            "Error: Tavily API key is not set. "
            "Run /tavily to save it, or set TAVILY_API_KEY in your environment."
        )

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


def execute_grep(
    pattern: str,
    path: str = ".",
    glob: str = "",
    case_insensitive: bool = False,
    max_results: int = 50,
) -> str:
    """
    Search file contents for a regex/text pattern (cross-platform).
    Returns matching lines as path:line:content.
    """
    if not pattern:
        return "Error: pattern is required."

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    root = Path(path or ".").expanduser()
    if not root.exists():
        return f"Error: path '{path}' does not exist."

    matches: List[str] = []
    files_searched = 0
    truncated = False
    max_results = max(1, min(int(max_results or 50), 200))

    def _should_skip_dir(name: str) -> bool:
        return name in SKIP_DIR_NAMES or name.startswith(".")

    def _iter_files() -> Any:
        if root.is_file():
            yield root
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            for filename in filenames:
                yield Path(dirpath) / filename

    try:
        for filepath in _iter_files():
            if truncated:
                break
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() in SKIP_FILE_SUFFIXES:
                continue
            if glob:
                try:
                    if not filepath.match(glob) and not Path(filepath.name).match(glob):
                        continue
                except Exception:
                    continue

            files_searched += 1
            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    rel = filepath
                    try:
                        rel = filepath.resolve().relative_to(Path.cwd().resolve())
                    except Exception:
                        rel = filepath
                    matches.append(f"{rel}:{line_no}:{line.rstrip()}")
                    if len(matches) >= max_results:
                        truncated = True
                        break

        if not matches:
            scope = f" in '{path}'" if path and path != "." else ""
            g = f" (glob={glob})" if glob else ""
            return f"No matches for /{pattern}/{scope}{g}. Searched {files_searched} file(s)."

        header = f"Found {len(matches)} match(es) in {files_searched} file(s)"
        if truncated:
            header += f" (truncated at {max_results})"
        return header + ":\n" + "\n".join(matches)
    except Exception as e:
        return f"Error executing grep: {e}"


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
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search the workspace for a text/regex pattern across files. "
                "Prefer this over bash grep/findstr for code discovery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex or plain text pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: current workspace)."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional filename glob filter, e.g. '*.py' or '*.ts'."
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matches to return (default: 50, max: 200)."
                    }
                },
                "required": ["pattern"]
            }
        }
    }
]
