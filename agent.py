import llm
from models import Models
from rich.traceback import install
from console import get_console

import sys

install(show_locals=True)

def main():

    agent = llm.Agent()
    
    # Print welcome message
    agent.console.print_welcome()
    
    while True:
        try:
            # Get user input with enhanced UI
            user_query = agent.console.get_user_input("Task")
            
            if user_query.lower() in ['exit', 'quit']:
                agent.console.print_system_message("Goodbye!", "Exit")
                break
            
            if not user_query.strip():
                continue
              
            if user_query == "/resume":
                old_sessions = agent.memory.load_old_sessions()
                if not old_sessions:
                    agent.console.print_system_message("No previous sessions found.", "Resume")
                    continue
                selected_session = agent.console.interactive_select(old_sessions)
                agent.console.clear_screen()
                old_chats = agent.memory.load_session_chat(selected_session.history_path)
                agent.memory.session = selected_session
                agent.console.print_welcome(selected_session.title, str(selected_session.workspace))
                agent.console.print_chat_history(old_chats)
                agent.console.print_system_message(f"Resumed session: {selected_session.title}")
                
                continue
                
            # Print user message
            agent.console.print_user_message(user_query)
            agent.console.print_separator()
            
            # Show loading indicator
            with agent.console.print_loading("Processing your request..."):
                if agent.memory.session is None:
                    session = agent.memory.init_session(user_query) 
                    agent.current_session = session  # type: ignore
                    agent.console.clear_screen()
                    agent.console.print_welcome(session.title, str(session.workspace))
                    agent.console.print_system_message("New Conversation started")
                    
                response = agent.send(user_query)
            
            # Print assistant response
            agent.console.print_separator()
            agent.console.print_assistant_message(response.message.content)  # type: ignore
            agent.console.print_separator()
            
        except KeyboardInterrupt:
            agent.console.print_error("\nOperation cancelled by user")
            break
        except Exception as e:
            agent.console.print_error(f"An error occurred: {str(e)}")
            agent.console.print_separator()


if __name__ == "__main__":
    main()