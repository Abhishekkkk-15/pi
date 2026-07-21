from dataclasses import dataclass
from enum import Enum
from pathlib import Path
class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

@dataclass
class Message:
    role: Role = Role.SYSTEM
    content: str = ""
    name:str|None = None
    
    def __post_init__(self):
        if self.name is None:
            del self.name
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for the Mistral API."""
        return {
            "role": self.role.value,   # Enum -> string
            "content": self.content
        }
        
        

@dataclass
class Session:
    id:str
    title:str
    workspace: Path
    history_path:Path

class Models(Enum):
    EMBEED = "mistral-embed"
    CHAT = "mistral-medium-latest"