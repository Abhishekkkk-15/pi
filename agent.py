import llm
from models import Models
from rich.traceback import install
install(show_locals=True)
def main():
    agent = llm.Agent(Models.CHAT)
    while True:
        user_qyery = input("Enter your task : ")
    
  
        print(agent.send(user_qyery).message.content) # type: ignore


if __name__ == "__main__":
    main()
