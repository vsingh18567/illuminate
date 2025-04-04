from abc import ABC
import base64
import json
import os
import subprocess
from loguru import logger
from openai import pydantic_function_tool
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessageToolCall
import nbformat as nbf


class Tool(BaseModel, ABC):

    def __call__(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


class LsTool(Tool):
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
            logger.error(f"Error listing directory: {e}")
            return {"files": [], "directories": [], "error": str(e)}


class GetFileInfoTool(Tool):
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
            logger.error(f"Error getting file info: {e}")
            return {
                "size": 0,
                "num_lines": 0,
                "error": str(e),
            }


class CatTool(Tool):
    """Read the contents of a file.

    If a file is too large, prefer creating Python scripts to get information from the file.

    Image and PDF files are not supported.

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
                if len(content) > 20_000:  # TODO: Make this configurable
                    return {
                        "content": "",
                        "error": "File is too large to read",
                    }
                return {"content": content}
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {"content": "", "error": str(e)}


class ViewPDFTool(Tool):
    """This tool gets user permission to upload a PDF file. If this tool is successfully called, the user will upload the requested PDF file.

    Invoking this tool is both useful and expensive. Don't do it unless it's needed for you to complete the task.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the PDF file to view")

    def __call__(self) -> dict:
        return {"success": True}


class ViewImageTool(Tool):
    """This tool gets user permission to upload an image file. If this tool is successfully called, the user will upload the requested image file.

    Invoking this tool is both useful and expensive. Don't do it unless it's needed for you to complete the task.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the image file to view")

    def __call__(self) -> dict:
        return {"success": True}


class WriteFileTool(Tool):
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
            logger.error(f"Error writing file: {e}")
            return {"success": False, "error": str(e)}


class DeleteFileTool(Tool):
    """Delete a file

    Only files within the current directory or its subdirectories can be deleted.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the file to delete")

    def __call__(self) -> dict:
        try:
            # Get absolute paths for comparison
            abs_path = os.path.abspath(self.path)
            current_dir = os.path.abspath(os.getcwd())

            # Check if the file is within the current directory or subdirectories
            if not abs_path.startswith(current_dir):
                return {
                    "success": False,
                    "error": "Cannot delete files outside the current directory",
                }

            # Check if the file exists
            if not os.path.isfile(abs_path):
                return {"success": False, "error": f"File not found: {self.path}"}

            os.remove(abs_path)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {"success": False, "error": str(e)}


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


class HtmlToPdfTool(Tool):
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
                    "--allow",
                    ".",
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
            logger.error(f"Error converting HTML to PDF: {e}")
            return {"success": False, "error": str(e)}


class JupyterNotebookCell(BaseModel):
    """A cell in a jupyter notebook

    Returns a JSON object with the following fields:
        cell_type: str
        source: str
    """

    cell_type: str = Field(..., description="The type of the cell")
    source: str = Field(..., description="The source code of the cell")


class CreateJupyterNotebookTool(Tool):
    """Create a jupyter notebook from the work done in a data science project.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(..., description="The path to the jupyter notebook to create")

    def __call__(self) -> dict:
        nb = nbf.v4.new_notebook()
        nb["cells"] = []
        try:
            with open(self.path, "w") as f:
                nbf.write(nb, f)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error creating notebook: {e}")
            return {"success": False, "error": str(e)}


class GetJupyterNotebookCellsTool(Tool):
    """Get the cells of a jupyter notebook

    Internally uses nbformat

    Returns a JSON object with the following fields:
        cells: list[JupyterNotebookCell]
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to get the cells of"
    )

    def __call__(self) -> dict:
        if not os.path.exists(self.path):
            return {"error": f"File not found: {self.path}"}
        # Load an existing notebook
        with open(self.path, "r", encoding="utf-8") as f:
            notebook = nbf.read(f, as_version=4)

        # Extract all cells
        cells = notebook["cells"]
        print("GetJupyterNotebookCellsTool cells", type(cells), str(cells))
        output = []
        for i, cell in enumerate(cells):
            output.append(
                JupyterNotebookCell(cell_type=cell["cell_type"], source=cell["source"])
            )

        return {"cells": output}


class AddJupyterNotebookCellsTool(Tool):
    """Add cells to an existing jupyter notebook

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

        with open(self.path, "r", encoding="utf-8") as f:
            notebook = nbf.read(f, as_version=4)
        for cell in self.cells:
            if cell.cell_type == "code":
                new_cell = nbf.v4.new_code_cell(cell.source)
            else:
                new_cell = nbf.v4.new_markdown_cell(cell.source)
            notebook["cells"].append(new_cell)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                nbf.write(notebook, f)
            return {"success": True}
        except Exception as e:
            logger.error(f"Error writing notebook: {e}")
            return {"success": False, "error": str(e)}


TOOLS: list[type[Tool]] = [
    LsTool,
    GetFileInfoTool,
    CatTool,
    WriteFileTool,
    DeleteFileTool,
    RunPythonTool,
    PipInstallTool,
    HtmlToPdfTool,
    ViewImageTool,
    ViewPDFTool,
]

JUPYTER_NOTEBOOK_TOOLS: list[type[Tool]] = TOOLS + [
    CreateJupyterNotebookTool,
    GetJupyterNotebookCellsTool,
    AddJupyterNotebookCellsTool,
]


TOOL_DEFINITIONS: list[ChatCompletionToolParam] = [
    pydantic_function_tool(tool) for tool in TOOLS
]
JUPYTER_NOTEBOOK_TOOL_DEFINITIONS: list[ChatCompletionToolParam] = [
    pydantic_function_tool(tool) for tool in JUPYTER_NOTEBOOK_TOOLS
]

FILE_REQUEST_TOOLS: set[str] = {ViewPDFTool.__name__, ViewImageTool.__name__}


def execute_tool_request_file(
    tool_call: ChatCompletionMessageToolCall,
) -> tuple[str, str | None]:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    shortened_args = ""
    for k, v in tool_args.items():
        if len(v) > 20:
            shortened_args += f"{k}: {v[:30]}...,"
        else:
            shortened_args += f"{k}: {v},"
    logger.info(f"Executing tool: {tool_name}({shortened_args})")

    for tool in JUPYTER_NOTEBOOK_TOOLS:
        if tool.__name__ == tool_name:
            if tool_name in FILE_REQUEST_TOOLS:
                return json.dumps({"success": True}), tool_args["path"]
            else:
                tool_instance = tool(**tool_args)
                return json.dumps(tool_instance()), None
    return json.dumps({"error": f"Tool {tool_name} not found"}), None
