import os
from loguru import logger
from pydantic import BaseModel, Field
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from illuminate.tools.base import Tool


class JupyterNotebookCell(BaseModel):
    """A cell in a jupyter notebook

    Returns a JSON object with the following fields:
        cell_type: str
        source: str
        output: list[str] | None
    """

    cell_type: str = Field(..., description="The type of the cell")
    source: str = Field(..., description="The source code of the cell")
    output: list[str] = Field(
        ..., description="The text outputs of the cell, if it has been executed"
    )


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


def cells_to_base_model(cells: list[nbf.NotebookNode]) -> list[dict]:
    formatted_cells = []

    for _, cell in enumerate(cells):
        outputs = []
        for output in cell["outputs"]:
            output_type = output["output_type"]
            if output_type == "execute_result":
                output_text = output["data"]["text/plain"]
            elif output_type == "display_data":
                output_text = output["data"]["text/plain"]
            elif output_type == "stream":
                output_text = output["text"]
            elif output_type == "error":
                output_text = str(output["traceback"])
            outputs.append(output_text)
        formatted_cells.append(
            {
                "cell_type": cell["cell_type"],
                "source": cell["source"],
                "output": outputs,
            }
        )
    return formatted_cells


class GetJupyterNotebookCellsTool(Tool):
    """Get the cells of a jupyter notebook

    Internally uses nbformat

    Returns a JSON object with the following fields:
        cells: list[dict]
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
        return {"cells": cells_to_base_model(cells)}


class AddJupyterNotebookCellsTool(Tool):
    """Add cells to an existing jupyter notebook.

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to add the cells to"
    )
    cells: list[JupyterNotebookCell] = Field(
        ...,
        description="The cells to add to the jupyter notebook. The cells should have no output.",
    )

    def __call__(self) -> dict:
        if not os.path.exists(self.path):
            return {"error": f"File not found: {self.path}"}

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


class RemoveLastJupyterNotebookCellTool(Tool):
    """Remove the last cell from a jupyter notebook

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to remove the last cell of"
    )

    def __call__(self) -> dict:
        if not os.path.exists(self.path):
            return {"error": f"File not found: {self.path}"}
        with open(self.path, "r", encoding="utf-8") as f:
            notebook = nbf.read(f, as_version=4)
        if len(notebook["cells"]) == 0:
            return {"error": "No cells to remove"}
        notebook["cells"].pop()
        with open(self.path, "w", encoding="utf-8") as f:
            nbf.write(notebook, f)
        return {"success": True}


class ExecuteJupyterNotebookCellTool(Tool):
    """Execute all the cells in a jupyter notebook and get back all the executed cells

    Returns a JSON object with the following fields:
        success: bool
        error: str | None
        cells: list[dict]
    """

    path: str = Field(
        ..., description="The path to the jupyter notebook to execute the cell of"
    )

    def __call__(self) -> dict:
        if not os.path.exists(self.path):
            return {"error": f"File not found: {self.path}"}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                notebook = nbf.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            path_to_execute = os.path.dirname(os.path.abspath(self.path))
            print(path_to_execute)
            out_cells = ep.preprocess(
                notebook, {"metadata": {"path": path_to_execute}}
            )[0]["cells"]
            return {"success": True, "cells": cells_to_base_model(out_cells)}
        except Exception as e:
            logger.error(f"Error executing notebook: {e}")
            return {"success": False, "error": str(e)}


JUPYTER_NOTEBOOK_TOOLS: list[type[Tool]] = [
    CreateJupyterNotebookTool,
    GetJupyterNotebookCellsTool,
    AddJupyterNotebookCellsTool,
    ExecuteJupyterNotebookCellTool,
    RemoveLastJupyterNotebookCellTool,
]
