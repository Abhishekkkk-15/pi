from mistralai.client import Mistral
import os
import prompts
from enum import Enum
from dotenv import load_dotenv
from  dataclasses import dataclass
from models import Role,Message,Models
load_dotenv()
from mistralai import extra

api_key = os.getenv("LLM_KEY")


class Agent:
    def __init__(self, model):
        self.model = model
        self.client = Mistral(api_key=api_key)
        self.prompt = prompts.Prompt()
        self.messages:list[Message]  =  [
            Message(role=Role.SYSTEM,content=self.prompt.prompts[0])
        ]
    def chat(self, messages:list[Message]):
        api_messages = [msg.to_dict() for msg in self.messages]
        res = self.client.chat.complete(
            model=self.model,
            messages=api_messages # type: ignore
        )
        return res.choices[0]
        
        
    def send(self, user_query):
        self.messages.append(Message(role=Role.USER, content=user_query))
        
        user_prompt = self.prompt.generate_user_prompt(self.messages)
        user_message = Message(role=Role.USER, content=user_prompt)

        # 4. Send to API (must be a list, even if only one message)
        api_messages = [user_message.to_dict()]
        return self.client.chat.complete(
            model=self.model,
            messages=api_messages # type: ignore
        )


