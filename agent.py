import llm
from models import Models
from rich.traceback import install
from console import get_console

import sys

install(show_locals=True)

def main():
    console = get_console()
    agent = llm.Agent()
    
    # Print welcome message
    console.print_welcome()
    
    while True:
        try:
            # Get user input with enhanced UI
            user_query = console.get_user_input("Task")
            
            if user_query.lower() in ['exit', 'quit']:
                console.print_system_message("Goodbye!", "Exit")
                break
            
            if not user_query.strip():
                continue
              
            if user_query == "/resume":
                old_sessions = agent.memory.load_old_sessions()
                if not old_sessions:
                    console.print_system_message("No previous sessions found.", "Resume")
                    continue
                selected_session = console.interactive_select(old_sessions)
                console.clear_screen()
                old_chats = agent.memory.load_session_chat(selected_session.history_path)
                agent.memory.session = selected_session
                console.print_welcome(selected_session.title, str(selected_session.workspace))
                console.print_chat_history(old_chats)
                console.print_system_message(f"Resumed session: {selected_session.title}")
                
                continue
                
            # Print user message
            console.print_user_message(user_query)
            console.print_separator()
            
            # Show loading indicator
            with console.print_loading("Processing your request..."):
                if agent.memory.session is None:
                    session = agent.memory.init_session(user_query) 
                    agent.current_session = session  # type: ignore
                    console.clear_screen()
                    console.print_welcome(session.title, str(session.workspace))
                    console.print_system_message("New Conversation started")
                    
                response = agent.send(user_query)
            
            # Print assistant response
            console.print_separator()
            console.print_assistant_message(response.message.content)  # type: ignore
            console.print_separator()
            
        except KeyboardInterrupt:
            console.print_error("\nOperation cancelled by user")
            break
        except Exception as e:
            console.print_error(f"An error occurred: {str(e)}")
            console.print_separator()


if __name__ == "__main__":
    main()