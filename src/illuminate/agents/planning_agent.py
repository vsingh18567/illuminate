from enum import Enum
from openai import OpenAI
from pydantic import BaseModel

from illuminate.agents.agent import Agent

from openai.types.chat import ParsedChatCompletionMessage


class PlanningAction(str, Enum):
    """
    An action that can be taken by the planning agent.
    """

    PLAN = "plan"
    USER_QUESTION = "user_question"
    DONE = "done"


class PlanningResponse(BaseModel):
    """
    A response from the planning agent.

    Set the action to PLAN if you are outputting a list of steps.
    Set the action to USER_QUESTION if you need to ask the user a question.
    Set the action to DONE if you are done with the project.
    """

    action: PlanningAction
    steps_description: list[str] | None = None
    user_question: str | None = None
    final_summary: str | None = None


class PlanningAgent(Agent):
    PLANNING_SYSTEM_PROMPT = """
    You are a smart, detail-oriented planning agent that is part of a team of agents that are building a data science project. The project should follow the traditional data science method of understanding the problem, inspecting and cleaning the data, conducting data analysis, doing modelling if necessary and producing a report of findings (using the HtmlToPdfTool). 

    You will be given a project description and a list of tools that can be used to build the project. You will need to plan the steps of the project. Your steps should be detailed and include the expected output of the step.

    The agents are smart and can deal with complex multi-step tasks. Each agent is responsible for a single step of the project.

    After the agent is done, another agent will review their work and provide feedback. Once both agents are happy with the step, they will give you a summary of the work they did. You can then update the plan if you need to. This process will repeat until the project is complete. When the project is complete, you will output the final summary of the project.

    Your response should only contain all the steps that still need to be completed - if the agent did not successfully complete a step, you should continue to include it in your response. 

    At any point, you can ask the user for clarifications on the project description or the steps.

    Set the PlanningResponse.action to plan if you are outputting a list of steps (each step should be its own string). Otherwise, set the action to user_question if you need to ask the user a question. Set the action to done if you are done with the project and are outputting a final summary.
    """

    def __init__(self, user_prompt: str):
        super().__init__(
            name="PlanningAgent",
            description="The planning agent is responsible for planning the steps of the project.",
            model="gpt-4o",
            system_prompt=PlanningAgent.PLANNING_SYSTEM_PROMPT,
            messages=[],
            response_model=PlanningResponse,
        )
        self.add_message(user_prompt, user=True)
        self.user_prompt = user_prompt
        self.plan = []

    def run(self, client: OpenAI) -> tuple[PlanningAction, list[str] | str]:
        response = super().query(client)
        parsed_response = response.parsed
        if parsed_response.action == PlanningAction.PLAN:
            self.plan = parsed_response.steps_description
            return parsed_response.action, self.plan
        elif parsed_response.action == PlanningAction.USER_QUESTION:
            if self.ask_user(response):
                return self.run(client)
            else:
                return None, None
        else:
            return parsed_response.action, parsed_response.final_summary

    def ask_user(self, message: ParsedChatCompletionMessage) -> bool:
        print(message.parsed.user_question)
        response = input()
        if response == "exit":
            return False
        self.add_message(response, user=True)
        return True
