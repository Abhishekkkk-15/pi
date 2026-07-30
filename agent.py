import argparse
import llm
from models import Models, Message, Role
from rich.traceback import install
from console import get_console
from interrupt import AgentInterrupted
from commands import Commands
import sys

install(show_locals=True)

def main():
    parser = argparse.ArgumentParser(description="PI - Python Agent Harness")
    parser.add_argument("--autonomous-risk", "-a", action="store_true", help="Allow agent to execute all tools automatically without asking for permission")
    args = parser.parse_args()

    agent = llm.Agent()
    commands = Commands(agent=agent)
    if args.autonomous_risk:
        agent.config.autonomous_risk = True

    # Print welcome message
    agent.console.print_welcome()
    if agent.config.autonomous_risk:
        agent.console.print_error("⚠️ AUTONOMOUS RISK MODE ACTIVE: All tool actions will execute without confirmation prompts.", "Autonomous Mode")
    
    while True:
        try:
            # Get user input with enhanced UI
            user_query = agent.console.get_user_input("Task")
            
            if user_query.lower() in ['exit', 'quit']:
                agent.console.print_system_message("Goodbye!", "Exit")
                break
            
            if not user_query.strip():
                continue
            continue_true = commands.router(user_query)  
            if continue_true:
                continue
            # if user_query == "/resume":
            #     old_sessions = agent.memory.load_old_sessions()
            #     if not old_sessions:
            #         agent.console.print_system_message("No previous sessions found.", "Resume")
            #         continue
            #     selected_session = agent.console.interactive_select(old_sessions)
            #     agent.console.clear_screen()
            #     old_chats = agent.memory.load_session_chat(selected_session.history_path, system_prompt=agent.prompt.raw_system_prompt)
            #     agent.memory.session = selected_session
            #     agent.console.print_welcome(selected_session.title, str(selected_session.workspace))
            #     agent.console.print_chat_history(old_chats)
            #     agent.console.print_system_message(f"Resumed session: {selected_session.title}")
                
            #     continue
                
            # Print user message once (prompt echo is erased in get_user_input)
            agent.console.print_user_message(user_query)
            agent.console.print_separator()
            
            # Show loading indicator
            with agent.console.print_loading("Processing your request... (ESC to stop)"):
                if agent.memory.session is None:
                    session = agent.memory.init_session(user_query, initial_messages=agent.memory.messages) 
                    agent.current_session = session  # type: ignore
                    agent._flush_pending_usage()
                    agent.console.clear_screen()
                    agent.console.print_welcome(session.title, str(session.workspace))
                    agent.console.print_system_message("New Conversation started")
                    # clear_screen wiped the User block — print it again
                    agent.console.print_user_message(user_query)
                    agent.console.print_separator()
                    
                response = agent.send(user_query)
            
            # Interrupted or LLM error already handled/persisted inside Agent.chat
            if response is None:
                agent.console.print_separator()
                agent.print_session_usage()
                continue

            # Print assistant response
            agent.console.print_separator()
            agent.console.print_assistant_message(response.message.content)  # type: ignore
            agent.console.print_separator()
            agent.print_session_usage()
            
        except KeyboardInterrupt:
            # At the prompt: exit. During a turn, Agent.chat already handled interrupt.
            agent.console.print_system_message("Goodbye!", "Exit")
            break
        except AgentInterrupted:
            agent.console.print_separator()
            agent.print_session_usage()
            continue
        except Exception as e:
            agent.console.print_error(f"An error occurred: {str(e)}")
            # Persist unexpected runtime errors into the active session history
            if agent.memory and agent.memory.session:
                err_msg = Message(
                    role=Role.ASSISTANT,
                    content=f"[Error] {type(e).__name__}: {e}",
                )
                agent.memory.messages.append(err_msg)
                agent.memory.write_to_jsonl(
                    agent.memory.session.history_path, [err_msg], mode="a"
                )
            agent.console.print_separator()
            agent.print_session_usage()


if __name__ == "__main__":
    main()
