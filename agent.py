import llm
from rich.traceback import install
install(show_locals=True)
def main():
    agent = llm.Agent(llm.Models.CHAT)
    while True:
        user_qyery = input("Enter your task : ")
    
  
        print(agent.send(user_qyery).choices[0].message.content)


if __name__ == "__main__":
    main()
