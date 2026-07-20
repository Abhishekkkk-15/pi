from models import Message
import xml.etree.ElementTree as ET
from pathlib import Path

class Prompt:
    prompts:list[str]
    def __init__(self):
        system_prompt:str = self.read_file_content("./prompts/SYSTEM.md")
        # system_prompt = self.read_file_content("SYSTEM.md")
        self.prompts = [system_prompt]
    
    def generate_user_prompt(self,messages: list[Message]) -> str:
        if not messages:
            return ""

        # Root element for structuring
        root = ET.Element("user_prompt")

        # 1. Past conversation history (all messages except the last)
        history_messages = messages[:-1]
        if history_messages:
            history_elem = ET.SubElement(root, "conversation_history")
            for msg in history_messages:
                msg_elem = ET.SubElement(history_elem, "message", role=msg.role.value)
                msg_elem.text = msg.content

        # 2. Current user turn (the last message in the list)
        current_msg = messages[-1]
        current_elem = ET.SubElement(root, "current_user_message")
        current_elem.text = current_msg.content

        # Format with indentation for readability
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
    
    def read_file_content(self,path:str) -> str:
        cwd = Path.cwd()
        print(cwd) 
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    