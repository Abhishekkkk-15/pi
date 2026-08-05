import json
from models import Message, Role
import uuid
from models import Session
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Union
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()


def generate_chat_id():
    return uuid.uuid4().hex


def is_development() -> bool:
    """True when ENV/ENVIRONMENT is development|dev (local package work)."""
    mode = (
        os.getenv("ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("enviroment")
        or ""
    ).strip().lower()
    return mode in ("development", "dev")


def get_data_root() -> Path:
    """
    Directory that holds auth.json and session folders (.pi-python).

    - Development (ENV=development|dev): next to this package
      (the pi-python repo root when running from source / editable install)
      e.g. D:\\python\\pi-python\\.pi-python

    - Installed / production: under the user's home directory
      Windows: C:\\Users\\<username>\\.pi-python
      Linux:   /home/<username>/.pi-python
      macOS:   /Users/<username>/.pi-python
    """
    if is_development():
        # memory.py lives at the project / package root
        return Path(__file__).resolve().parent / ".pi-python"
    return Path.home() / ".pi-python"


class _DataRoot:
    """Descriptor so `Memory.root / "auth.json"` always resolves at access time."""

    def __get__(self, obj, objtype=None) -> Path:
        return get_data_root()


class Memory:
    messages: list[Message]
    session: Union[Session, None]
    root = _DataRoot()

    def __init__(self):
        self.messages = []
        self.session = None

    def init_session(self, title: str, initial_messages: list[Message] | None = None) -> Session:
        id = generate_chat_id()
        # Workspace = project the user launched the agent from (always cwd)
        workspace = Path.cwd()
        # Sessions/auth live under the env-specific data root
        data_root = get_data_root()
        history_path = data_root / id
        history_path.mkdir(parents=True, exist_ok=True)
        conversation_jsonl = history_path / "conversation_history.jsonl"
        conversation_jsonl.touch()
        session = Session(
            id=id,
            title=title,
            workspace=workspace,
            history_path=conversation_jsonl,
        )
        self.session = session
        self.write_to_json(history_path / "metadata.json", session)
        if initial_messages:
            self.write_to_jsonl(conversation_jsonl, initial_messages, mode="a")
        return session

    def load_old_sessions(self) -> list[Session]:
        memory_path = get_data_root()

        if not memory_path.exists():
            return []

        sessions = []
        for folder in memory_path.iterdir():
            if not folder.is_dir():
                continue
            # Skip non-session files/dirs (e.g. auth.json lives at root)
            metadata_file = folder / "metadata.json"
            if not metadata_file.exists():
                continue
            try:
                with open(metadata_file, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    session = Session(
                        id=data.get("id", folder.name),
                        title=data.get("title", folder.name),
                        workspace=Path(data["workspace"]) if "workspace" in data else Path.cwd(),
                        history_path=Path(data["history_path"])
                        if "history_path" in data
                        else folder / "conversation_history.jsonl",
                        permissions=data.get(
                            "permissions",
                            {
                                "allow_all": False,
                                "allowed_tools": [],
                                "allowed_targets": {},
                            },
                        ),
                        prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(data.get("completion_tokens", 0) or 0),
                        total_tokens=int(data.get("total_tokens", 0) or 0),
                        cached_tokens=int(data.get("cached_tokens", 0) or 0),
                        estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0) or 0.0),
                        compaction_summary=str(data.get("compaction_summary", "") or ""),
                        compacted_until=int(data.get("compacted_until", 0) or 0),
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

    @staticmethod
    def write_to_json(path: Union[str, Path], data: Any) -> None:
        file_path = Path(path)
        payload = asdict(data) if is_dataclass(data) else data  # type: ignore

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=4, default=str)

        except TypeError as e:
            raise TypeError(
                f"Failed to serialize data for {file_path.name}: {e}"
            ) from e
        except OSError as e:
            raise RuntimeError(f"Could not write to {file_path}: {e}")

    @staticmethod
    def read_from_json(path: Union[str, Path]) -> Any:
        """Reads and parses data from a JSON file."""
        file_path = Path(path)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None
        except OSError:
            return None

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
                if line:
                    data = json.loads(line)

                    role_val = data.get("role", "system")
                    role_enum = Role.from_val(role_val)

                    message = Message(
                        role=role_enum,
                        content=data.get("content", ""),
                        name=data.get("name", None),
                        tool_calls=data.get("tool_calls", None),
                        tool_call_id=data.get("tool_call_id", None),
                        reasoning_content=data.get("reasoning_content", None),
                    )
                    messages.append(message)

        return messages
