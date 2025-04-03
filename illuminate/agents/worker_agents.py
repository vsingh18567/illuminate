from illuminate.agents.agent import Agent
from illuminate.tools import JUPYTER_NOTEBOOK_TOOL_DEFINITIONS, TOOL_DEFINITIONS
from loguru import logger
from openai import OpenAI


class WorkerAgent(Agent):
    WORKER_AGENT_SYSTEM_PROMPT = """
    You are a smart, detail-oriented worker agent that is part of a team of agents that are building a data science project. You are responsible for completing a step of the project to the best of your abilities. Don't take shortcuts. Be certain that your approach is the best possible approach.

    You will be given a step description, and a summary of any previous steps. Upon completing the step, you will summarize all the work you did in enough detail so that another agent can review your work and provide feedback. When your reviewer is happy with your work, you will output a complete summary of your work.

    Prefer creating files to communicate information with other agents.
    """

    def __init__(
        self, id: int, user_prompt: str, work_so_far: str, step_description: str
    ):
        super().__init__(
            name=f"WorkerAgent{id}",
            description="The worker agent is responsible for completing a step of the project.",
            model="gpt-4o",
            system_prompt=WorkerAgent.WORKER_AGENT_SYSTEM_PROMPT,
            messages=[],
            tools=TOOL_DEFINITIONS,
        )
        logger.info(f"Worker agent initialised to solve: {step_description}")
        self.add_message(
            f"The user prompt is: {user_prompt}\n\n The work done so far is: {work_so_far}\n\n Your job is to complete the following step: {step_description}",
            user=True,
        )

    def run(self, client: OpenAI) -> str:
        response = super().query(client)
        logger.info(f"{self.name}: {response.content}")
        return response.content


class IpynbAgent(Agent):
    IPYNB_AGENT_SYSTEM_PROMPT = """
    You are an agent that is responsible for creating a jupyter notebook from the work done in a data science project. 

    You will be given the user prompt, and the work done so far. You will need to create a jupyter notebook cell by cell using the tools provided.

    ONLY add code cells to the notebook that uses code already done in the project.
    """

    def __init__(self, user_prompt: str, work_so_far: str):
        super().__init__(
            name="Ipynb Agent",
            description="The ipynb agent is responsible for creating a jupyter notebook from the work done in a data science project.",
            model="gpt-4o",
            system_prompt=IpynbAgent.IPYNB_AGENT_SYSTEM_PROMPT,
            messages=[],
            tools=JUPYTER_NOTEBOOK_TOOL_DEFINITIONS,
        )

        self.add_message(
            f"The user prompt is: {user_prompt}\n\n The work done so far is: {work_so_far}",
            user=True,
        )

    def run(self, client: OpenAI) -> str:
        response = super().query(client)
        return response.content
