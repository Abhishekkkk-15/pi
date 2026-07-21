
from dotenv import load_dotenv
import os
from models import Models
load_dotenv()
class Config:
    llm_api_key:str|None 
    is_dev: bool = True
    model:Models = Models.CHAT 
    def __init__(self):
        api_key = os.getenv("LLM_KEY")
        if not api_key:
            raise RuntimeError("LLM API key is Required!")
        is_dev = os.getenv("enviroment") 
        self.llm_api_key =  api_key
        self.is_dev = True if is_dev else False
        
        
