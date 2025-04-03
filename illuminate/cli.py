import json
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from illuminate.agents.agent_system import AgentSystem
from illuminate.tools import TOOL_DEFINITIONS, execute_tool
from loguru import logger


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
