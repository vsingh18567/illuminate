from dataclasses import dataclass
from enum import Enum
import json

from openai import NOT_GIVEN, NotGiven, OpenAI, beta
from openai.types.chat import ChatCompletionMessageToolCall, ParsedChatCompletionMessage
from tools import TOOL_DEFINITIONS, execute_tool
from loguru import logger
from pydantic import BaseModel


@dataclass
class Agent:
    name: str
    description: str
    model: str
    system_prompt: str
    messages: list[dict]
    response_model: BaseModel | NotGiven = NOT_GIVEN

    def __post_init__(self):
        self.messages.append({"role": "system", "content": self.system_prompt})

    def query(self, client: OpenAI) -> ParsedChatCompletionMessage:
        completion = beta.chat.completions.parse(
            model=self.model,
            messages=self.messages,
            tools=TOOL_DEFINITIONS,
            response_format=self.response_model,
        )
        self.add_message(completion.choices[0].message, user=False)
        if completion.choices[0].message.tool_calls:
            logger.info(f"{self.name} is calling tools")
            for tool_call in completion.choices[0].message.tool_calls:
                self.call_tool(tool_call)
            return self.query(client)
        logger.info(f"{self.name} is not calling tools")

        return completion.choices[0].message

    def add_message(self, message: dict, user: bool):
        if user:
            self.messages.append({"role": "user", "content": message})
            return
        if message.parsed is not None:
            parsed = message.parsed
            content = parsed if isinstance(parsed, str) else parsed.model_dump_json()
            self.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        if message.tool_calls is not None:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append(tool_call.model_dump())
                self.messages.append(
                    {"role": "assistant", "content": [], "tool_calls": tool_calls}
                )

    def call_tool(self, tool_call: ChatCompletionMessageToolCall):
        result = execute_tool(tool_call)
        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )


class PlanningAction(str, Enum):
    """
    An action that can be taken by the planning agent.
    """

    PLAN = "plan"
    USER_INTERACTION = "user_interaction"


class PlanningResponse(BaseModel):
    """
    A response from the planning agent.

    Set the action to PLAN if you are outputting a list of steps.
    Otherwise, set the action to USER_INTERACTION.
    """

    action: PlanningAction
    steps_description: list[str] | None = None
    user_interaction: str | None = None


class PlanningAgent(Agent):
    PLANNING_SYSTEM_PROMPT = """
    You are a planning agent that is part of a team of agents that are building a data science project.

    You will be given a project description and a list of tools that can be used to build the project. You will need to plan the steps of the project. Your steps should be detailed and include the expected output of the step (e.g. a file name, a paragraph of text, a chart, etc.). Your agents are smart and can deal with complex multi-step tasks.

    After the agent does their step, you will need to review the results and update the plan accordingly. Your response should only contain all the steps that still need to be completed - if the agent did not successfully complete a step, you should continue to include it in your response.

    At any point, you can ask the user for clarifications on the project description or the steps. 
    """

    def __init__(self, user_prompt: str):
        super().__init__(
            name="Planning Agent",
            description="The planning agent is responsible for planning the steps of the project.",
            model="gpt-4o",
            system_prompt=PlanningAgent.PLANNING_SYSTEM_PROMPT,
            messages=[],
            response_model=PlanningResponse,
        )
        self.add_message(user_prompt, user=True)
        self.user_prompt = user_prompt
        self.plan = []

    def run(self, client: OpenAI) -> list[str]:
        response = super().query(client)
        parsed_response = response.parsed
        if parsed_response.action == PlanningAction.PLAN:
            self.plan = parsed_response.steps_description
            return self.plan
        elif parsed_response.action == PlanningAction.USER_INTERACTION:
            if self.ask_user(response):
                return self.run(client)
            else:
                return None

    def ask_user(self, message: ParsedChatCompletionMessage) -> bool:
        response = input()
        if response == "exit":
            return False
        self.add_message(response, user=True)
        return True


class WorkerAgent(Agent):
    WORKER_AGENT_SYSTEM_PROMPT = """
    You are a worker agent that is part of a team of agents that are building a data science project. You are responsible for completing a step of the project.

    You will be given a step description, and a summary of any previous steps. Upon completing the step, return a summary of what you did in enough detail that the next agent can continue from where you left off.

    Do not get overenthusiastic and do work that goes beyond the step description.
    """

    def __init__(self, user_prompt: str, work_so_far: str, step_description: str):
        super().__init__(
            name="Worker Agent",
            description="The worker agent is responsible for completing a step of the project.",
            model="gpt-4o",
            system_prompt=WorkerAgent.WORKER_AGENT_SYSTEM_PROMPT,
            messages=[],
        )
        self.add_message(
            f"The user prompt is: {user_prompt}\n\n The work done so far is: {work_so_far}\n\n Your job is to complete the following step: {step_description}",
            user=True,
        )

    def run(self, client: OpenAI) -> str:
        response = super().query(client)
        return response.content


class AgentSystem:

    def __init__(self, user_prompt: str, client: OpenAI):
        self.client = client
        self.planning_agent = PlanningAgent(user_prompt)
        self.worker_agents = []
        self.work_done = []

    def add_work_done(self, work_done: str):
        agent_id = len(self.work_done) + 1
        self.work_done.append(f"Agent {agent_id} did the following work: {work_done}")

    def run(self):
        while True:
            plan = self.planning_agent.run(self.client)
            if not plan:
                break
            assert len(plan) > 0, "Plan is empty"
            worker = WorkerAgent(
                self.planning_agent.user_prompt,
                "\n".join(self.work_done),
                plan[0],
            )
            self.add_work_done(worker.run(self.client))
            self.planning_agent.add_message(self.work_done[-1], user=True)
