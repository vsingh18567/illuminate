from openai import OpenAI
from dotenv import load_dotenv
from illuminate.agents.agent_system import AgentSystem


def main():
    load_dotenv()

    client = OpenAI()
    with open("prompt.txt", "r") as f:
        user_prompt = f.read()

    agent_system = AgentSystem(
        user_prompt,
        client,
    )
    agent_system.run()


if __name__ == "__main__":
    main()
