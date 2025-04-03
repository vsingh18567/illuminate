import base64
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


class GetFileInfoTool(BaseModel):
    """Get information about a file

    Returns a JSON object with the following fields:
        size: int (bytes)
        num_lines: int
    """

    path: str = Field(..., description="The path to the file to get information about")

    def __call__(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return {
                    "size": os.path.getsize(self.path),
                    "num_lines": len(f.readlines()),
                }
        except Exception as e:
            return {
                "size": 0,
                "num_lines": 0,
                "error": str(e),
            }


class CatTool(BaseModel):
    """Read the contents of a file.

    If a file is too large, prefer creating Python scripts to get information from the file.

    Returns a JSON object with the following fields:
        content: str
        error: str | None
    """

    path: str = Field(..., description="The path to the file to read")

    def __call__(self) -> dict:
        try:
            extension = os.path.splitext(self.path)[-1]
            if extension == ".png" or extension == ".jpg" or extension == ".jpeg":
                return {
                    "content": "",
                    "error": "File is an image, cannot be viewed",
                }
            elif extension == ".csv":
                return {
                    "content": "",
                    "error": "File is a CSV, use scripts to get information from the file.",
                }
            with open(self.path, "r") as f:
                content = f.read()
                if len(content) > 50_000:
                    return {
                        "content": "",
                        "error": "File is too large to read",
                    }
                return {"content": content}
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


class CreateJupyterNotebookTool(BaseModel):
    """Create a jupyter notebook from the work done in a data science project.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the jupyter notebook to create")

    def __call__(self) -> dict:
        write_file_tool = WriteFileTool(
            path=self.path,
            content=json.dumps(
                {
                    "cells": [],
                    "metadata": {"language_info": {"name": "python"}},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
        )
        return write_file_tool()


class RunPythonTool(BaseModel):
    """Run an existing Python script

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


class HtmlToPdfTool(BaseModel):
    """Convert an HTML file to a PDF file

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    html_file: str = Field(
        ..., description="The path to the HTML file to convert to PDF"
    )
    pdf_file: str = Field(
        ..., description="The path to the PDF file to save the result to"
    )

    def __call__(self) -> dict:
        try:
            result = subprocess.run(
                [
                    "wkhtmltopdf",
                    self.html_file,
                    self.pdf_file,
                ],  # TODO: Add installation somewhere
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return {"success": True}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}


class JupyterNotebookCell(BaseModel):
    """A cell in a jupyter notebook

    Returns a JSON object with the following fields:
        cell_type: str
        source: str
    """

    cell_type: str = Field(..., description="The type of the cell")
    source: str = Field(..., description="The source code of the cell")


class GetJupyterNotebookCellsTool(BaseModel):
    """Get the cells of a jupyter notebook

    Returns a JSON object with the following fields:
        cells: list[JupyterNotebookCell]
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to get the cells of"
    )

    def __call__(self) -> dict:
        with open(self.path, "r") as f:
            notebook = json.load(f)
            return {
                "cells": [
                    JupyterNotebookCell(
                        cell_type=cell["cell_type"], source=cell["source"]
                    )
                    for cell in notebook["cells"]
                ]
            }


class AddJupyterNotebookCellsTool(BaseModel):
    """Add cells to a jupyter notebook

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to add the cells to"
    )
    cells: list[JupyterNotebookCell] = Field(
        ..., description="The cells to add to the jupyter notebook"
    )

    def __call__(self) -> dict:
        with open(self.path, "r") as f:
            notebook = json.load(f)
        notebook["cells"].extend(self.cells)
        with open(self.path, "w") as f:
            json.dump(notebook, f)
        return {"success": True}


TOOLS: list[BaseModel] = [
    LsTool,
    GetFileInfoTool,
    CatTool,
    WriteFileTool,
    RunPythonTool,
    PipInstallTool,
    HtmlToPdfTool,
]

JUPYTER_NOTEBOOK_TOOLS: list[BaseModel] = [
    CreateJupyterNotebookTool,
    GetJupyterNotebookCellsTool,
    AddJupyterNotebookCellsTool,
]

TOOL_DEFINITIONS: list[dict] = [pydantic_function_tool(tool) for tool in TOOLS]
JUPYTER_NOTEBOOK_TOOL_DEFINITIONS: list[dict] = TOOL_DEFINITIONS + [
    pydantic_function_tool(tool) for tool in JUPYTER_NOTEBOOK_TOOLS
]


def execute_tool(tool_call: dict) -> dict:
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
            tool_instance = tool(**tool_args)
            return json.dumps(tool_instance())
    return {"error": f"Tool {tool_name} not found"}
