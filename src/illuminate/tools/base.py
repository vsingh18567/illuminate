from abc import ABC
from pydantic import BaseModel


class Tool(BaseModel, ABC):

    def __call__(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

