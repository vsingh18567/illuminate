import os
import subprocess
from loguru import logger
from pydantic import Field
from illuminate.tools.base import Tool


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
            if not os.path.exists(os.path.dirname(self.path)):
                os.makedirs(os.path.dirname(self.path))
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


FILE_TOOLS: list[type[Tool]] = [
    LsTool,
    GetFileInfoTool,
    CatTool,
    WriteFileTool,
    DeleteFileTool,
]
