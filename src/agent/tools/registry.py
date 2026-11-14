from typing import Callable, Dict, List, Optional
from ollama import Tool
from src.agent.tools.base import BaseTool


class ToolRegistry:
    """
    Central registry holding BaseTool instances.
    Constructs Ollama tool specifications, dispatches executions by name,
    and dynamically merges custom dynamic tools.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self._tools: Dict[str, BaseTool] = {}
        self._functions: Dict[str, Callable] = {}
        self._external_specs: Dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._functions[tool.name] = tool.run

    def register_function(
        self, name: str, func: Callable, spec: Optional[Tool] = None
    ) -> None:
        self._functions[name] = func
        if spec:
            self._external_specs[name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._functions.pop(name, None)
        self._external_specs.pop(name, None)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        return self._functions.get(name)

    def get_specs(self) -> List[Tool]:
        specs = [tool.to_spec() for tool in self._tools.values()]
        specs.extend(self._external_specs.values())
        return specs

    def get_functions(self) -> Dict[str, Callable]:
        return dict(self._functions)

    def merge_custom(
        self, custom_tools: List[Tool], custom_funcs: Dict[str, Callable]
    ) -> "ToolRegistry":
        merged = ToolRegistry(list(self._tools.values()))
        for name, spec in self._external_specs.items():
            merged.register_function(name, self._functions[name], spec)
        for name, func in custom_funcs.items():
            merged.register_function(name, func)
        for spec in custom_tools:
            spec_name = (
                spec.function.name
                if hasattr(spec, "function") and hasattr(spec.function, "name")
                else getattr(spec, "name", "")
            )
            if spec_name:
                merged._external_specs[spec_name] = spec
        return merged

    def dispatch(self, name: str, **kwargs) -> str:
        if name in self._functions:
            try:
                return str(self._functions[name](**kwargs))
            except Exception as e:
                return f"Error: {e}"
        return f"Error: unknown tool '{name}'"


def create_default_registry(settings=None, client=None) -> ToolRegistry:
    from src.agent.settings import SETTINGS
    from src.agent.tools.builder import ToolBuilderTool
    from src.agent.tools.calculator import CalculatorTool
    from src.agent.tools.cache import CacheTool
    from src.agent.tools.cli import CliTool
    from src.agent.tools.memory import MemoryTool
    from src.agent.tools.planner import PlannerTool
    from src.agent.tools.web import web_fetch_spec, web_search_spec

    cfg = settings or SETTINGS
    ollama_client = client or cfg.OLLAMA_CLIENT

    registry = ToolRegistry(
        [
            CalculatorTool(),
            ToolBuilderTool(),
            MemoryTool(),
            CliTool(),
            PlannerTool(),
            CacheTool(),
        ]
    )

    if hasattr(ollama_client, "web_search"):
        registry.register_function(
            "web_search",
            ollama_client.web_search,
            web_search_spec if cfg.OLLAMA_WEB_SEARCH else None,
        )
    if hasattr(ollama_client, "web_fetch"):
        registry.register_function(
            "web_fetch",
            ollama_client.web_fetch,
            web_fetch_spec if cfg.OLLAMA_WEB_SEARCH else None,
        )

    return registry
