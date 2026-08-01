"""Conversation compaction (WIP).

Use shared token helpers from ``tokenizer``::

    from tokenizer import count_messages, count_history_jsonl, load_tokenizer
"""

from __future__ import annotations

import json

def jsonl_to_clean_transcript(jsonl_data: str) -> str:
    """
    Converts a JSONL string into a clean, human-readable transcript.
    Strips out system prompt bloat and formats tool calls/results clearly.
    """
    formatted_lines = []

    for line in jsonl_data.strip().splitlines():
        if not line.strip():
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = msg.get("role", "").upper()
        content = msg.get("content", "") or ""
        
        # 1. Handle SYSTEM messages
        if role == "SYSTEM":
            # Strip or heavily shorten huge system prompts to save context space
            formatted_lines.append("[SYSTEM PROMPT INCLUDED]")
            continue

        # 2. Handle USER messages
        if role == "USER":
            formatted_lines.append(f"USER:\n{content.strip()}\n")
            continue

        # 3. Handle ASSISTANT messages (and their Tool Calls)
        if role == "ASSISTANT":
            assistant_block = []
            if content.strip():
                assistant_block.append(f"ASSISTANT:\n{content.strip()}")
            
            # Extract tool calls made by the assistant
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool in msg["tool_calls"]:
                    fn_name = tool.get("function", {}).get("name", "unknown_tool")
                    fn_args = tool.get("function", {}).get("arguments", "{}")
                    assistant_block.append(f"[CALL TOOL: {fn_name}({fn_args})]")
            
            if assistant_block:
                formatted_lines.append("\n".join(assistant_block) + "\n")
            continue

        # 4. Handle TOOL / FUNCTION response messages
        if role in ("TOOL", "FUNCTION"):
            # Optional: Truncate excessively long tool outputs (e.g. giant logs/file dumps)
            max_len = 1000
            if len(content) > max_len:
                content = content[:max_len] + f"\n... [Truncated {len(content) - max_len} characters]"

            formatted_lines.append(f"TOOL RESULT:\n{content.strip()}\n")
            continue

    return "\n---\n".join(formatted_lines)

class Compaction:
    
    def compact():
        
        return 
