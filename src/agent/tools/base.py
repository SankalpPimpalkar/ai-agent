from abc import ABC, abstractmethod
from typing import Any, Dict
from ollama import Tool


class BaseTool(ABC):
    """
    Abstract base class for all tools in the system.
    Follows Interface Segregation and Liskov Substitution principles.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The public tool name exposed to Ollama."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """The tool description explaining its purpose and usage."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema dictionary of accepted tool parameters."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Executes the tool with keyword arguments and returns a string response."""
        ...

    def to_spec(self) -> Tool:
        """Constructs the standard Ollama Tool specification object."""
        return Tool(
            type="function",
            function=Tool.Function(
                name=self.name,
                description=self.description.strip(),
                parameters=Tool.Function.Parameters(
                    type=self.parameters.get("type", "object"),
                    properties=self.parameters.get("properties", {}),
                    required=self.parameters.get("required", []),
                ),
            ),
        )
