from loguru import logger
from openai import OpenAI

from illuminate.agents.planning_agent import PlanningAgent
from illuminate.agents.review_agent import ReviewAgent
from illuminate.agents.worker_agents import IpynbAgent, WorkerAgent


class AgentSystem:

    def __init__(self, user_prompt: str, client: OpenAI):
        self.client = client
        self.planning_agent = PlanningAgent(user_prompt)
        self.worker_agents = []
        self.work_done = []
        self.step_id = 0

    def add_work_done(self, agent_name: str, work_done: str):
        self.work_done.append(f"{agent_name} did the following work: {work_done}")

    def worker_step(self, plan: list[str]):
        self.step_id += 1
        worker = WorkerAgent(
            self.step_id,
            self.planning_agent.user_prompt,
            "\n".join(self.work_done),
            plan[0],
        )

        reviewer = ReviewAgent(
            self.step_id,
            self.planning_agent.user_prompt,
            plan[0],
            worker.run(self.client),
        )

        while True:
            review = reviewer.run(self.client)
            if review.passed:
                worker.add_message(
                    f"The reviewer has passed the work. Providing a summary of all the work you did.",
                    user=True,
                )
                logger.info(f"Agent {self.step_id} has passed the work.")
                self.add_work_done(worker.name, worker.run(self.client))
                self.planning_agent.add_message(self.work_done[-1], user=True)
                break
            else:
                logger.info(f"Feedback: {review.feedback}")
                worker.add_message(
                    f"The reviewer has given the following feedback: {review.feedback}",
                    user=True,
                )
                reviewer.add_message(
                    f"The worker has made the following changes: {worker.run(self.client)}",
                    user=True,
                )

    def run(self):
        while True:
            plan = self.planning_agent.run(self.client)
            if not plan:
                break
            self.worker_step(plan)
        ipynb_agent = IpynbAgent(
            user_prompt=self.planning_agent.user_prompt,
            work_so_far="\n".join(self.work_done),
        )
        ipynb_agent.run(self.client)
