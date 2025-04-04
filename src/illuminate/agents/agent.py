import base64
from dataclasses import dataclass
from enum import Enum
import json
import os

from openai import NOT_GIVEN, NotGiven, OpenAI, beta
from openai.types.chat import ChatCompletionMessageToolCall, ParsedChatCompletionMessage
from illuminate.tools import (
    JUPYTER_NOTEBOOK_TOOL_DEFINITIONS,
    JUPYTER_NOTEBOOK_TOOLS,
    TOOL_DEFINITIONS,
    TOOLS,
    execute_tool_request_file,
)
from loguru import logger
from pydantic import BaseModel

LOG_FOLDER = "illuminate_logs"


@dataclass
class Agent:
    name: str
    description: str
    model: str
    system_prompt: str
    messages: list[dict]
    tools: list[BaseModel]
    response_model: BaseModel | NotGiven = NOT_GIVEN

    def __post_init__(self):
        self._add_message({"role": "system", "content": self.system_prompt})

    def query(self, client: OpenAI) -> ParsedChatCompletionMessage:
        completion = beta.chat.completions.parse(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            response_format=self.response_model,
        )
        self.add_message(completion.choices[0].message, user=False)
        if completion.choices[0].message.tool_calls:
            logger.info(f"{self.name} is calling tools")
            for tool_call in completion.choices[0].message.tool_calls:
                self.call_tool(tool_call, client)
            return self.query(client)
        logger.info(f"{self.name} is done.")

        return completion.choices[0].message

    def _add_message(self, message: dict):
        self.messages.append(message.copy())
        if not os.path.exists(LOG_FOLDER):
            os.makedirs(LOG_FOLDER)
        if "content" in message:
            message["content"] = (
                message["content"][:1000] + "..."
                if len(message["content"]) > 1000
                else message["content"]
            )
        with open(f"{LOG_FOLDER}/{self.name}.txt", "a") as f:
            f.write(json.dumps(message))
            f.write(f"\n{'-' * 20}\n")

    def add_message(self, message: ParsedChatCompletionMessage, user: bool):
        if user:
            self._add_message({"role": "user", "content": message})
            return
        if message.parsed is not None:
            parsed = message.parsed
            content = parsed if isinstance(parsed, str) else parsed.model_dump_json()
            self._add_message(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        if message.tool_calls is not None:
            tool_calls = []
            for tool_call in message.tool_calls:
                tool_calls.append(tool_call.model_dump())
            self._add_message(
                {"role": "assistant", "content": [], "tool_calls": tool_calls}
            )

    def _add_file_request(self, file_path: str, client: OpenAI):
        file_type = file_path.split(".")[-1]
        if file_type == "pdf":
            file = client.files.create(file=open(file_path, "rb"), purpose="user_data")
            self._add_message(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_id": file.id,
                            },
                        },
                    ],
                }
            )
        elif file_type in ["png", "jpg", "jpeg"]:
            with open(file_path, "rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")
            self._add_message(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{file_type};base64,{image}",
                            },
                        },
                    ],
                }
            )

    def call_tool(self, tool_call: ChatCompletionMessageToolCall, client: OpenAI):
        result, file_path = execute_tool_request_file(tool_call)
        self._add_message(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
        if file_path is not None:
            self._add_file_request(file_path, client)
