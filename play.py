import json
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents import AgentSystem
from tools import TOOL_DEFINITIONS, execute_tool
from loguru import logger

load_dotenv()

client = OpenAI()

agent_system = AgentSystem(
    "Help me investigate the health of US citizens compared to other countries. I have given you data in 'data.csv'. I want you to use the tools available to you to investigate the data and provide me with a report.",
    client,
)
agent_system.run()
