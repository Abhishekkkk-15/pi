import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json
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


def execute_bash(command: str, timeout: int = 60) -> str:
    """Executes a bash command and returns stdout + stderr."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = res.stdout + res.stderr
        return output.strip() if output else "[Command finished with no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing bash command: {str(e)}"


def dispatch_tool_call(tool_name:str, function_arguments:str):
    args = json.loads(function_arguments)
    if tool_name == "read":
        return execute_read(args["path"])
    
    elif tool_name == "write":
        return execute_write(args["path"], args["content"])
    
    elif tool_name == "edit":
        return execute_edit(args["path"], args["edits"])
    
    elif tool_name == "bash":
        return execute_bash(args["command"])
    
    else:
        return f"Unknown tool: {tool_name}"
    

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
            "description": "Run shell/bash commands (ls, git, pytest, grep, find, etc.). Use this for system actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command string to execute."
                    }
                },
                "required": ["command"]
            }
        }
    }
]