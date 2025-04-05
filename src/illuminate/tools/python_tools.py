import subprocess
from loguru import logger
from pydantic import Field
from illuminate.tools.base import Tool


class RunPythonTool(Tool):
    """Run an existing Python script

    Do not use this tool to try and print out the contents of a large file.

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


class PipInstallTool(Tool):
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
            logger.error(f"Error installing package: {result.stderr}")
            return {"success": False, "error": result.stderr}


PYTHON_TOOLS: list[type[Tool]] = [
    RunPythonTool,
    PipInstallTool,
]
