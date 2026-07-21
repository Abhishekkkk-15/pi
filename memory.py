import json
from models import Message
import uuid
from models import Session
from pathlib import Path
from dataclasses import asdict
def generate_chat_id():
    return  uuid.uuid4().hex

    

class Memory:
    session: Session
    message: list[Message]
    
    def init_session(self,title:str) -> Session:
        id = generate_chat_id()
        title = title
        workspace = Path.cwd()
        history_path = workspace / ".memory" / id
        history_path.mkdir(parents=True, exist_ok=True)
        session = Session(
            id=id,
            title=title,
            workspace=workspace,
            history_path=history_path
        )
        with open(history_path/"metadata.json","w", encoding="utf-8") as file:
            json.dump(asdict(session), file, indent=4, default=str)
                
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
                                history_path=Path(data["history_path"]) if "history_path" in data else folder
                            )
                            sessions.append(session)
                    except (json.JSONDecodeError, KeyError, OSError):
                        continue  # Skip corrupted folders

        return sessions
        
    def load_session(self,id:str) -> Session | None:
        if not id :
            return
        
        # return Session()
    
    def write_to_jsonl(self) -> None:
        pass
    
    def write_to_json(self) -> None:
        pass
    
    def read_from_jsonl(self) -> None:
        pass
    
    def read_from_json(self) -> None:
        pass