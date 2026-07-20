import llm
from models import Models
from rich.traceback import install
from console import get_console
import sys

install(show_locals=True)

def main():
    console = get_console()
    agent = llm.Agent(Models.CHAT)
    
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
                
            # Print user message
            console.print_user_message(user_query)
            console.print_separator()
            
            # Show loading indicator
            with console.print_loading("Processing your request..."):
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