from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Message:
    role: Role = Role.SYSTEM
    content: str = ""
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for the Mistral API."""
        return {
            "role": self.role.value,   # Enum -> string
            "content": self.content
        }

class Models(Enum):
    EMBEED = "mistral-embed"
    CHAT = "mistral-medium-3-5"