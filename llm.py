from mistralai.client import Mistral
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from models import Role,Message
from tools import TOOLS, dispatch_tool_call

import json
load_dotenv()

api_key = os.getenv("LLM_KEY")


class Agent:
    def __init__(self, model):
        self.model = model
        self.client = Mistral(api_key=api_key)
        self.prompt = prompts.Prompt()
        self.messages:list[Message]  =  [
            Message(role=Role.SYSTEM,content=self.prompt.prompts[0])
        ]
    def chat(self, user_query:str):
        
        self.messages.append(Message(role=Role.USER, content=user_query))
        
        user_prompt = self.prompt.generate_user_prompt(self.messages)
        user_message = Message(role=Role.USER, content=user_prompt)

        api_messages = [user_message.to_dict()]
        while True:
            
            res = self.client.chat.complete(
                model=self.model,
                messages=api_messages, # type: ignore
                tools=TOOLS # type: ignore
            )
            llm_res = res.choices[0]
            api_messages.append({
                "role":"assistant",
                "content":llm_res.message.model_dump(exclude_unset=True),  # type: ignore
                "tool_calls": llm_res.message.tool_calls   # type: ignore
            })
            
            if llm_res.message.tool_calls:  # type: ignore
                for tool in llm_res.message.tool_calls:  # type: ignore
                    tool_name = tool.function.name
                    tool_arguments = tool.function.arguments
                    print(f"calling tool : {tool_name} with arguments : {tool_arguments}")
                    fn_output = dispatch_tool_call(tool_name,tool_arguments)  # type: ignore
                    api_messages.append({
                        "role":"tool",
                        "content":fn_output,
                        "tool_call_id":tool.id
                    })
            else:
                return res.choices[0]        
            # api_messages.append()
            

        
        
    def send(self, user_query):
     
        return self.chat(user_query)


