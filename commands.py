from llm import Agent
from typing import Any, Callable, Dict, List
import inspect
class Commands:
    
    def __init__(self,agent:Agent) -> None:
        self.agent = agent
        self.commands = self.get_methods()
    def router(self,command:str):
        for s in self.commands:
            cmp = s.get(command)
            if cmp != None:
               return cmp()
                
    from typing import Any, Callable, Dict


    def get_methods(self) -> List[Dict[str, Callable[..., Any]]]:
        """Returns a list of single-entry dictionaries mapping method name to bound method."""
        methods: List[Dict[str, Callable[..., Any]]] = []

        for attr_name in dir(self):
            if (
                attr_name.startswith("__")
                or attr_name.startswith("get")
                or attr_name == "router"
            ):
                continue

            attr = getattr(self, attr_name)
            if inspect.ismethod(attr):
                methods.append({f"/{attr_name}": attr})

        return methods
    
    def resume(self) -> bool:
        agent = self.agent
        old_sessions = agent.memory.load_old_sessions()
        if not old_sessions:
            agent.console.print_system_message("No previous sessions found.", "Resume")
            return True
        selected_session = agent.console.interactive_select(old_sessions)
        agent.console.clear_screen()
        old_chats = agent.memory.load_session_chat(selected_session.history_path, system_prompt=agent.prompt.raw_system_prompt)
        agent.memory.session = selected_session
        agent.console.print_welcome(selected_session.title, str(selected_session.workspace))
        agent.console.print_chat_history(old_chats)
        agent.console.print_system_message(f"Resumed session: {selected_session.title}")
        
        return True

    def login(self) -> bool:
        
        agent = self.agent
        api_key = self.agent.console.get_api_key(self.agent.config.provider)
        if not api_key:
            return True
        if not self.agent.memory.root:
            print(self.agent.memory.root)
            return True
        try:        
            auth_root = self.agent.memory.root / "auth.json"
            json = agent.memory.read_from_json(auth_root)
            if not json:
                json = {"credentials":{"api_key" : api_key}}
            json["credentials"]["api_key"] = api_key
            agent.memory.write_to_json(auth_root,json)
            agent.config.api_key = api_key
            return True
        except FileNotFoundError:    
            return True
        except:
            return True