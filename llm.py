from mistralai.client import Mistral
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from models import Role,Message
from tools import TOOLS
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
        res = self.client.chat.complete(
            model=self.model,
            messages=api_messages, # type: ignore
            tools=TOOLS # type: ignore
        )
        
        return res.choices[0]
        
        
    def send(self, user_query):
     
        return self.chat(user_query)


