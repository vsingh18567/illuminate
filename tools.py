import json
import os
import subprocess
from loguru import logger
from openai import pydantic_function_tool
from pydantic import BaseModel, Field


class LsTool(BaseModel):
    """List the contents of a directory

    Returns a JSON object with the following fields:
        files: list[str]
        directories: list[str]
        error: str | None
    """

    path: str = Field(..., description="The path to list the contents of")

    def __call__(self) -> dict:
        files = []
        directories = []
        try:
            for item in os.listdir(self.path):
                if os.path.isfile(os.path.join(self.path, item)):
                    files.append(item)
                elif os.path.isdir(os.path.join(self.path, item)):
                    directories.append(item)
            return {"files": files, "directories": directories}
        except Exception as e:
            return {"files": [], "directories": [], "error": str(e)}


class CatTool(BaseModel):
    """Read the contents of a file

    Returns a JSON object with the following fields:
        content: str
        error: str | None
    """

    path: str = Field(..., description="The path to the file to read")

    def __call__(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return {"content": f.read()}
        except Exception as e:
            return {"content": "", "error": str(e)}


class WriteFileTool(BaseModel):
    """Write to a file, creating it if it doesn't exist and overwriting it if it does

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the file to write to")
    content: str = Field(..., description="The content to write to the file")

    def __call__(self) -> dict:
        try:
            with open(self.path, "w") as f:
                f.write(self.content)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


class RunPythonTool(BaseModel):
    """Run a Python script

    Returns a JSON object with the following fields:
        stdout: str
        stderr: str
        returncode: int
    """

    script: str = Field(..., description="Path to the Python script to run")

    def __call__(self) -> dict:
        result = subprocess.run(["python", self.script], capture_output=True, text=True)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }


class PipInstallTool(BaseModel):
    """Install a Python package

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    package: str = Field(..., description="The package to install")

    def __call__(self) -> dict:
        result = subprocess.run(
            ["pip", "install", self.package], capture_output=True, text=True
        )
        if result.returncode == 0:
            return {"success": True}
        else:
            return {"success": False, "error": result.stderr}


TOOLS: list[BaseModel] = [LsTool, CatTool, WriteFileTool, RunPythonTool, PipInstallTool]
TOOL_DEFINITIONS: list[dict] = [pydantic_function_tool(tool) for tool in TOOLS]


def execute_tool(tool_call: dict) -> dict:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    logger.info(f"Executing tool: {tool_name}")

    for tool in TOOLS:
        if tool.__name__ == tool_name:
            return json.dumps(tool(**tool_args)())
    return {"error": f"Tool {tool_name} not found"}
