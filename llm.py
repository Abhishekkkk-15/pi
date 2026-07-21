from mistralai.client import Mistral
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from models import Role,Message, Session
from tools import TOOLS, dispatch_tool_call
from console import get_console
from config import Config
from memory import Memory
from dataclasses import asdict
load_dotenv()

api_key = os.getenv("LLM_KEY")


class Agent:
    def __init__(self):
        config = Config()
        self.config = config
       
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
        
        api_messages = [f.to_dict() for f in self.memory.messages]
        while True:
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
            api_messages.append(chat_msg.to_dict())
            if self.memory and self.memory.session:
                self.memory.write_to_jsonl(self.memory.session.history_path, [chat_msg], mode="a")
            
            if llm_res.message.tool_calls:  # type: ignore
                for tool in llm_res.message.tool_calls:  # type: ignore
                    tool_name = tool.function.name
                    tool_arguments = tool.function.arguments
                    self.console.print_tool_call(tool_name, tool_arguments)  # type: ignore
                    fn_output = dispatch_tool_call(tool_name, tool_arguments)  # type: ignore
                    self.console.print_tool_result(fn_output)
                    tool_msg = Message(
                        role=Role.TOOL,
                        name=tool_name,
                        content=fn_output,
                        tool_call_id=tool.id
                    )
                    self.memory.messages.append(tool_msg)
                    api_messages.append(tool_msg.to_dict())
                    if self.memory and self.memory.session:
                        self.memory.write_to_jsonl(self.memory.session.history_path, [tool_msg], mode="a")
            else:
                return res.choices[0]        
            # api_messages.append()
            
    def send(self, user_query):
     
        return self.chat(user_query)


