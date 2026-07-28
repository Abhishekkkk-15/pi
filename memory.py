import json
from models import Message,Role
import uuid
from models import Session
from pathlib import Path
from dataclasses import asdict,is_dataclass
from typing import Any, Union
from enum import Enum

def generate_chat_id():
    return  uuid.uuid4().hex

    

class Memory:
    messages: list[Message]
    session: Union[Session, None]

    def __init__(self):
        self.messages = []
        self.session = None
    
    def init_session(self, title: str, initial_messages: list[Message] = None) -> Session:
        id = generate_chat_id()
        workspace = Path.cwd()
        history_path = workspace / ".memory" / id
        history_path.mkdir(parents=True, exist_ok=True)
        conversation_jsonl = history_path / "conversation_history.jsonl"
        conversation_jsonl.touch()
        session = Session(
            id=id,
            title=title,
            workspace=workspace,
            history_path=conversation_jsonl
        )
        self.session = session
        self.write_to_json(history_path / "metadata.json", session)    
        if initial_messages:
            self.write_to_jsonl(conversation_jsonl, initial_messages, mode="a")
        return session
    
    def load_old_sessions(self) -> list[Session]:
        memory_path = Path.cwd() / ".memory"

        if not memory_path.exists():
            return []

        sessions = []
        for folder in memory_path.iterdir():
            if folder.is_dir():
                metadata_file = folder / "metadata.json"

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as file:
                            data = json.load(file)

                            # Reconstruct the Session object
                            session = Session(
                                id=data.get("id", folder.name),
                                title=data.get("title", folder.name),
                                workspace=Path(data["workspace"]) if "workspace" in data else Path.cwd(),
                                history_path=Path(data["history_path"]) if "history_path" in data else folder / "conversation_history.jsonl",
                                permissions=data.get("permissions", {
                                    "allow_all": False,
                                    "allowed_tools": [],
                                    "allowed_targets": {}
                                })
                            )
                            sessions.append(session)
                    except (json.JSONDecodeError, KeyError, OSError):
                        continue  # Skip corrupted folders

        return sessions
        
    def load_session_chat(self, path: Path, system_prompt: str = "") -> list[Message]:
        if not path or not path.exists():
            return []
        old_chat = self.read_from_jsonl(path=path)
        if not old_chat or old_chat[0].role != Role.SYSTEM:
            if system_prompt:
                sys_msg = Message(role=Role.SYSTEM, content=system_prompt)
                old_chat.insert(0, sys_msg)
        self.messages = old_chat
        return old_chat
        # return Session()
    
    def write_to_json(self, path: Union[str, Path], data: Any) -> None:
     """Writes data to a JSON file. Automatically converts dataclasses."""
     payload = asdict(data) if is_dataclass(data) else data #type: ignore

     with open(path, "w", encoding="utf-8") as file:
         json.dump(payload, file, indent=4, default=str)


    def read_from_json(self, path: Union[str, Path]) -> Any:
        """Reads and parses data from a JSON file."""
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


    def write_to_jsonl(
        self, path: Union[str, Path], data_list: list[Any], mode: str = "w"
    ) -> None:
        """
        Writes or appends a list of objects/dicts to a JSONL (JSON Lines) file.
        Use mode='a' to append to an existing file.
        """
        def default_serializer(obj: Any) -> Any:
            if isinstance(obj, Role):
                return obj.value
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, Path):
                return str(obj)
            return str(obj)

        with open(path, mode, encoding="utf-8") as file:
            for item in data_list:
                if hasattr(item, "to_dict"):
                    payload = item.to_dict()
                elif is_dataclass(item):
                    payload = asdict(item)
                else:
                    payload = item
                file.write(json.dumps(payload, default=default_serializer) + "\n")


    def read_from_jsonl(self, path: Union[str, Path]) -> list[Message]:
        messages = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    data = json.loads(line)

                    role_val = data.get("role", "system")
                    role_enum = Role.from_val(role_val)

                    message = Message(
                        role=role_enum,
                        content=data.get("content", ""),
                        name=data.get("name", None),
                        tool_calls=data.get("tool_calls", None),
                        tool_call_id=data.get("tool_call_id", None),
                    )
                    messages.append(message)

        return messages[-15:]

