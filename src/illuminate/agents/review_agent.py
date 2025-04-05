from openai import OpenAI
from pydantic import BaseModel

from illuminate.agents.agent import Agent


class ReviewResponse(BaseModel):
    """
    A response from the review agent.

    Set the passed to True if the work is good.
    Otherwise, set the passed to False and provide feedback on what needs to be improved.
    """

    passed: bool
    feedback: str | None


class ReviewAgent(Agent):
    REVIEW_SYSTEM_PROMPT = """
    You are a smart, detail-oriented review agent that is part of a team of agents that are building a data science project. The project has high standards in terms of thoroughness and optimality of the solution. Don't accept mediocre work. 

    You will be given the user's initial prompt, a worker's task and the work done by the worker. The worker has done the task in the current working directory.

    You will need to review the work done by the worker and provide feedback on what needs to be improved. The worker does not need to complete the entire user's request - just the task they have been assigned.

    Use the tools provided to review the work done by the worker (including checking the files they created).

    If the work is good, set the passed to True.
    Otherwise, set the passed to False and provide feedback on what needs to be improved.
    """

    def __init__(self, id: int, user_prompt: str, worker_task: str, worker_work: str):
        super().__init__(
            name=f"ReviewAgent{id}",
            description="Review the work done by the worker and provide feedback on what needs to be improved.",
            model="gpt-4o",
            system_prompt=ReviewAgent.REVIEW_SYSTEM_PROMPT,
            response_model=ReviewResponse,
            messages=[],
        )
        self.add_message(
            f"User prompt: {user_prompt}\n\nWorker task: {worker_task}\n\nWorker work: {worker_work}",
            user=True,
        )

    def run(self, client: OpenAI) -> ReviewResponse:
        response = super().query(client)
        return response.parsed
