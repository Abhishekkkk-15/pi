from llm import Agent

class Commands:
    
    def __init__(self,agent:Agent) -> None:
        self.agent = agent
        self.commands = [{"/resume":self.resume}]
    def router(self,command:str):
        for s in self.commands:
            cmp = s.get(command)
            if cmp != None:
               return cmp()
                

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
