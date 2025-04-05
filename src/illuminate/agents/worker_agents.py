from illuminate.agents.agent import Agent
from loguru import logger
from openai import OpenAI


class WorkerAgent(Agent):
    WORKER_AGENT_SYSTEM_PROMPT = """
    You are a smart, detail-oriented worker agent that is part of a team of agents that are building a data science project. You are responsible for completing a step of the project to the best of your abilities. Don't take shortcuts. Be certain that your approach is the best possible approach.

    You will be given a step description, and a summary of any previous steps. Upon completing the step, you will summarize all the work you did in enough detail so that another agent can review your work and provide feedback. Make sure your summary describes what you did, why you did it (unless it's obvious), and where evidence of your work can be found. When your reviewer is happy with your work, you will output a complete summary of your work. The summary should be less than 200 words.

    Using the tools provided, prefer creating (and adding onto existing) jupyter notebooks instead of python scripts wherever it is appropriate. All tools run from the current directory (the directory with the prompt.txt file). Do not create files in the illuminate_logs directory.

    Keep in mind that a future data scientist will be building on your work.
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
