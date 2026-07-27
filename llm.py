from mistralai.client import Mistral
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from models import Role,Message, Session
from tools import TOOLS
from console import get_console
from config import Config
from memory import Memory
from dataclasses import asdict
load_dotenv()
import json
from tools import *
api_key = os.getenv("LLM_KEY")


def sanitize_api_messages(raw_messages: list[dict]) -> list[dict]:
    """
    Sanitizes message history to guarantee Mistral API message ordering rules:
    1. Every assistant message with tool_calls must be followed by matching tool response messages for each tool_call_id.
    2. Missing tool responses are filled with a placeholder response so tool_call counts match.
    3. Orphan tool response messages (without a preceding matching assistant tool call) are removed.
    """
    sanitized: list[dict] = []
    i = 0
    n = len(raw_messages)

    while i < n:
        msg = raw_messages[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls") or []
            expected_ids = []
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    expected_ids.append(tc_id)

            sanitized.append(msg)
            i += 1

            tool_responses_by_id = {}
            while i < n and raw_messages[i].get("role") == "tool":
                tool_msg = raw_messages[i]
                t_id = tool_msg.get("tool_call_id")
                if t_id:
                    tool_responses_by_id[t_id] = tool_msg
                i += 1

            for t_id in expected_ids:
                if t_id in tool_responses_by_id:
                    sanitized.append(tool_responses_by_id[t_id])
                else:
                    sanitized.append({
                        "role": "tool",
                        "content": "Tool execution was interrupted.",
                        "tool_call_id": t_id
                    })
        elif role == "tool":
            i += 1
        else:
            sanitized.append(msg)
            i += 1

    return sanitized


class Agent:
    def __init__(self):
        config = Config()
        self.config = config
        self.console = get_console() 
        self.client = Mistral(api_key=api_key)
        self.prompt = prompts.Prompt()
        self.memory:Memory = Memory() 
        self.memory.messages = [Message(role=Role.SYSTEM, content=self.prompt.prompts[0])]
        self.console = get_console()
      
    def chat(self, user_query: str):
        user_msg = Message(role=Role.USER, content=user_query)
        self.memory.messages.append(user_msg)
        if self.memory and self.memory.session:
            self.memory.write_to_jsonl(self.memory.session.history_path, [user_msg], mode="a")
        
        while True:
            api_messages = sanitize_api_messages([f.to_dict() for f in self.memory.messages])
            res = self.client.chat.complete(  # type: ignore
                model=self.config.model, # type: ignore
                messages=api_messages, # type: ignore
                tools=TOOLS # type: ignore
            )
            llm_res = res.choices[0]
            chat_msg = Message(
                role=Role.ASSISTANT,
                content=llm_res.message.content or "",  # type: ignore
                tool_calls=llm_res.message.tool_calls   # type: ignore
            )
            self.memory.messages.append(chat_msg)
            if self.memory and self.memory.session:
                self.memory.write_to_jsonl(self.memory.session.history_path, [chat_msg], mode="a")
            
            if llm_res.message.tool_calls:  # type: ignore
                for tool in llm_res.message.tool_calls:  # type: ignore
                    tool_name = tool.function.name
                    tool_arguments = tool.function.arguments
                    self.console.print_tool_call(tool_name, tool_arguments)  # type: ignore
                    try:
                        fn_output = self.dispatch_tool_call(tool_name, tool_arguments)  # type: ignore
                    except Exception as e:
                        fn_output = f"Error executing tool {tool_name}: {str(e)}"
                    self.console.print_tool_result(fn_output)
                    tool_msg = Message(
                        role=Role.TOOL,
                        name=tool_name,
                        content=fn_output,
                        tool_call_id=tool.id
                    )
                    self.memory.messages.append(tool_msg)
                    if self.memory and self.memory.session:
                        self.memory.write_to_jsonl(self.memory.session.history_path, [tool_msg], mode="a")
            else:
                return res.choices[0]        
            # api_messages.append()
            
    def send(self, user_query):
     
        return self.chat(user_query)

    
    def dispatch_tool_call(self,tool_name:str, function_arguments:str):
        args = json.loads(function_arguments)
        if tool_name == "read":
            if not self.console.confirm_permission(f"Agent want to read {args["path"]}"):
                return "User permission deined"
            return execute_read(args["path"])

        elif tool_name == "write":
            if not self.console.confirm_permission(f"Agent want to {tool_name} {args["path"]}"):
                return "User permission deined"
            return execute_write(args["path"], args["content"])

        elif tool_name == "edit":
            if not self.console.confirm_permission(f"Agent want to {tool_name} {args["path"]}"):
                return "User permission deined"
            return execute_edit(args["path"], args["edits"])

        elif tool_name == "bash":
            cmd = args.get("command", "")
            timeout = args.get("timeout", 30)
            is_bg = args.get("is_background", False)
            bg_note = " (background)" if is_bg else ""
            if not self.console.confirm_permission(f"Agent wants to run {tool_name}{bg_note}: {cmd}"):
                return "User permission denied"
            return execute_bash(cmd, timeout=timeout, is_background=is_bg)

        else:
            return f"Unknown tool: {tool_name}"


