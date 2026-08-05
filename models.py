from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"
    ASSISTANT = "assistant"

    @classmethod
    def from_val(cls, val: Any) -> "Role":
        if isinstance(val, Role):
            return val
        if isinstance(val, str):
            clean = val.split(".")[-1].lower()
            for member in cls:
                if member.value == clean or member.name.lower() == clean:
                    return member
        return cls.SYSTEM

@dataclass
class Message:
    role: Role = Role.SYSTEM
    content: str = ""
    name:str|None = None
    tool_calls:list[Any]|None = None
    tool_call_id:str|None = None
    reasoning_content:str|None = None
    
    def __post_init__(self):
        if self.name is None and hasattr(self, "name"):
            del self.name
            
        if self.tool_call_id is None and hasattr(self, "tool_call_id"):
            del self.tool_call_id
            
        if self.tool_calls is None and hasattr(self, "tool_calls"):
            del self.tool_calls

        if self.reasoning_content is None and hasattr(self, "reasoning_content"):
            del self.reasoning_content
            
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict for the Mistral API."""
        role_str = self.role.value if isinstance(self.role, Role) else str(self.role)
        if isinstance(role_str, str) and "." in role_str:
            role_str = role_str.split(".")[-1].lower()
            
        data: dict[str, Any] = {
            "role": role_str,
            "content": self.content
        }
        if getattr(self, "name", None) is not None:
            data["name"] = self.name

        if getattr(self, "reasoning_content", None) is not None:
            data["reasoning_content"] = self.reasoning_content

        tool_calls_val = getattr(self, "tool_calls", None)
        if tool_calls_val is not None:
            formatted = []
            for tc in tool_calls_val:
                if hasattr(tc, "model_dump"):
                    formatted.append(tc.model_dump(exclude_none=True))
                elif isinstance(tc, dict):
                    formatted.append(tc)
                else:
                    formatted.append(tc)
            data["tool_calls"] = formatted

        tool_call_id_val = getattr(self, "tool_call_id", None)
        if tool_call_id_val is not None:
            data["tool_call_id"] = tool_call_id_val
        elif role_str == "tool":
            data["tool_call_id"] = "call_default"

        return data
        
        

from dataclasses import dataclass, field

@dataclass
class Session:
    id: str
    title: str
    workspace: Path
    history_path: Path
    permissions: dict = field(default_factory=lambda: {
        "allow_all": False,
        "allowed_tools": [],
        "allowed_targets": {}
    })
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Compaction: summary of messages[1:compacted_until]; recent tail stays raw
    compaction_summary: str = ""
    compacted_until: int = 0

class Models(Enum):
    EMBEED = "mistral-embed"
    CHAT = "mistral-large-latest"
    NVIDIA_LLAMA = "meta/llama-3.1-70b-instruct"
    NVIDIA_MISTRAL = "mistralai/mistral-7b-instruct-v0.3"
    NVIDIA_NEMOTRON = "nvidia/nemotron-3-ultra-550b-a55b"