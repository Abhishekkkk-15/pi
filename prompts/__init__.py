from models import Message
import xml.etree.ElementTree as ET
from importlib import resources
from pathlib import Path
from typing import Dict, Optional

class Prompt:
    def __init__(self):
        pass

    @property
    def raw_system_prompt(self) -> str:
        return self.get_system_prompt()

    @property
    def prompts(self) -> list[str]:
        return [self.get_system_prompt()]

    def get_system_prompt(
        self,
        active_skills: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        context_files: Optional[list[Dict[str, str]]] = None,
        selected_tools: Optional[list[str]] = None,
        tool_snippets: Optional[Dict[str, str]] = None,
        prompt_guidelines: Optional[list[str]] = None,
    ) -> str:
        """
        Builds the system prompt dynamically matching pi's system prompt architecture.
        Includes available tools, guidelines, project context (AGENTS.md), active skills,
        current date, and active working directory.
        """
        from datetime import datetime

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        effective_cwd = str(cwd if cwd is not None else Path.cwd()).replace("\\", "/")

        # 1. Base tools list
        tools = selected_tools or ["read", "bash", "edit", "write"]
        default_snippets = {
            "read": "Read file contents (supports text and images).",
            "bash": "Execute terminal commands.",
            "edit": "Make targeted file edits via exact string replacements.",
            "write": "Create new files or overwrite existing ones.",
            "grep": "Search workspace for text/regex patterns.",
            "web_search": "Perform web search for documentation/info.",
        }
        snippets = tool_snippets or default_snippets
        visible_tools = [name for name in tools if name in snippets]
        tools_list = (
            "\n".join(f"- {name}: {snippets[name]}" for name in visible_tools)
            if visible_tools
            else "(none)"
        )

        # 2. Dynamic guidelines
        guidelines_list: list[str] = []
        guidelines_set: set[str] = set()

        def add_guideline(g: str):
            g_clean = g.strip()
            if g_clean and g_clean not in guidelines_set:
                guidelines_set.add(g_clean)
                guidelines_list.append(g_clean)

        if "bash" in tools and not any(t in tools for t in ["grep", "find", "ls"]):
            add_guideline("Use bash for file operations like ls, rg, find")

        for g in prompt_guidelines or []:
            add_guideline(g)

        add_guideline("Be concise in your responses")
        add_guideline("Show file paths clearly when working with files")

        guidelines_formatted = "\n".join(f"- {g}" for g in guidelines_list)

        # 3. Base Prompt Assembly
        prompt = (
            "You are an expert coding assistant operating inside pi, a coding agent harness. "
            "You help users by reading files, executing commands, editing code, and writing new files.\n\n"
            f"Available tools:\n{tools_list}\n\n"
            "In addition to the tools above, you may have access to other custom tools depending on the project.\n\n"
            f"Guidelines:\n{guidelines_formatted}"
        )

        # 4. Project Context Injection (<project_context>)
        ctx_files = list(context_files or [])
        # Auto-discover AGENTS.md in current directory if context_files wasn't explicitly passed
        if not context_files:
            agents_md = Path(effective_cwd) / "AGENTS.md"
            if agents_md.is_file():
                try:
                    ctx_files.append({
                        "path": "AGENTS.md",
                        "content": agents_md.read_text(encoding="utf-8")
                    })
                except Exception:
                    pass

        if ctx_files:
            prompt += "\n\n<project_context>\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for ctx in ctx_files:
                file_path = ctx.get("path", "")
                content = ctx.get("content", "")
                prompt += f'<project_instructions path="{file_path}">\n{content}\n</project_instructions>\n\n'
            prompt += "</project_context>\n"

        # 5. Skills Section (if read tool is available)
        has_read = "read" in tools
        if has_read and active_skills:
            prompt += "\n\n### Active Skills\n\nThe following specialized skills have been loaded for this task:\n"
            for name, content in active_skills.items():
                prompt += f"\n#### Skill: {name}\n{content}\n"

        # 6. Metadata Anchoring (Date + CWD at bottom)
        prompt += f"\n\nCurrent date: {date_str}"
        prompt += f"\nCurrent working directory: {effective_cwd}"

        return prompt

    def generate_user_prompt(self, messages: list[Message]) -> str:
        if not messages:
            return ""

        root = ET.Element("user_prompt")

        history_messages = messages[:-1]
        if history_messages:
            history_elem = ET.SubElement(root, "conversation_history")
            for msg in history_messages:
                msg_elem = ET.SubElement(history_elem, "message", role=msg.role.value)
                msg_elem.text = msg.content

        current_msg = messages[-1]
        current_elem = ET.SubElement(root, "current_user_message")
        current_elem.text = current_msg.content

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
    