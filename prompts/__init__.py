from models import Message
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

class Prompt:
    prompts: list[str]
    raw_system_prompt: str

    def __init__(self):
        prompt_path = Path(__file__).parent / "SYSTEM.md"
        if not prompt_path.exists():
            prompt_path = Path("./prompts/SYSTEM.md")
        self.raw_system_prompt = prompt_path.read_text(encoding="utf-8")
        self.prompts = [self.raw_system_prompt]

    def get_system_prompt(self, active_skills: Optional[Dict[str, str]] = None) -> str:
        """
        Returns the system prompt, inserting active skills right after 
        the Runtime / Execution Rules section.
        """
        if not active_skills:
            return self.raw_system_prompt

        skills_md = "\n\n### Active Skills\n\nThe following specialized skills have been loaded for this task:\n"
        for name, content in active_skills.items():
            skills_md += f"\n#### Skill: {name}\n{content}\n"

        # Split around Problem-Solving Protocol (which follows Execution Rules)
        protocol_marker = "### 5. Problem-Solving Protocol"
        if protocol_marker in self.raw_system_prompt:
            parts = self.raw_system_prompt.split(protocol_marker, 1)
            return f"{parts[0].strip()}\n{skills_md}\n{protocol_marker}{parts[1]}"
        
        exec_marker = "### 4. Execution Rules"
        if exec_marker in self.raw_system_prompt:
            parts = self.raw_system_prompt.split(exec_marker, 1)
            return f"{parts[0]}{exec_marker}{parts[1].strip()}\n{skills_md}"

        return f"{self.raw_system_prompt.strip()}\n{skills_md}"

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
    