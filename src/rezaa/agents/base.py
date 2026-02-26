"""Base agent abstract class."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseAgent(ABC):
    """Abstract base class for all analysis agents."""

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Run analysis and return structured output."""
        ...

    def validate_output(self, output: BaseModel) -> bool:
        """Validate that the output conforms to its schema."""
        try:
            output.model_validate(output.model_dump())
            return True
        except Exception:
            return False

    def to_json(self, output: BaseModel) -> str:
        """Serialize output to JSON string."""
        return output.model_dump_json(indent=2)

    def from_json(self, json_str: str, model_class: type[BaseModel]) -> BaseModel:
        """Deserialize JSON string to model instance."""
        return model_class.model_validate_json(json_str)
