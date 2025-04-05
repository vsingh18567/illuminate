import json

from loguru import logger
from illuminate.tools.file_tools import FILE_TOOLS, ViewPDFTool, ViewImageTool
from illuminate.tools.ipynb_tools import JUPYTER_NOTEBOOK_TOOLS
from illuminate.tools.python_tools import PYTHON_TOOLS
from openai import pydantic_function_tool
from pydantic import BaseModel
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageToolCall

TOOLS = FILE_TOOLS + JUPYTER_NOTEBOOK_TOOLS + PYTHON_TOOLS
FILE_REQUEST_TOOLS = {ViewPDFTool.__name__, ViewImageTool.__name__}

TOOL_DEFINITIONS: list[ChatCompletionToolParam] = [
    pydantic_function_tool(tool) for tool in TOOLS
]


class ToolRequestResponse(BaseModel):
    tool_result: str
    requested_file: str | None


def execute_tool(
    tool_call: ChatCompletionMessageToolCall,
) -> ToolRequestResponse:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    shortened_args = ""
    for k, v in tool_args.items():
        if len(v) > 20:
            shortened_args += f"{k}: {v[:30]}...,"
        else:
            shortened_args += f"{k}: {v},"
    logger.info(f"Executing tool: {tool_name}({shortened_args})")

    for tool in TOOLS:
        if tool.__name__ == tool_name:
            if tool_name in FILE_REQUEST_TOOLS:
                return ToolRequestResponse(
                    tool_result=json.dumps({"success": True}),
                    requested_file=tool_args["path"],
                )
            else:
                tool_instance = tool(**tool_args)
                return ToolRequestResponse(
                    tool_result=json.dumps(tool_instance()),
                    requested_file=None,
                )
    return ToolRequestResponse(
        tool_result=json.dumps({"error": f"Tool {tool_name} not found"}),
        requested_file=None,
    )
