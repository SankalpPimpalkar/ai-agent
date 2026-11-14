from typing import Callable, Dict, List
from ollama import Tool
from src.agent.settings import SETTINGS

from src.agent.tools.base import BaseTool
from src.agent.tools.builder import (
    ToolBuilderTool,
    load_custom_tools,
    tool_builder,
    tool_builder_spec,
)
from src.agent.tools.calculator import (
    CalculatorTool,
    calculator_tool,
    calculator_tool_spec,
)
from src.agent.tools.cache import CacheTool, cache_tool, cache_tool_spec
from src.agent.tools.cli import (
    CliCommandValidator,
    CliFileOperations,
    CliPathGuard,
    CliTool,
    cli_tool,
    cli_tool_spec,
    safe_resolve_path,
    _safe_path,
)
from src.agent.tools.memory import (
    MemoryFileStore,
    MemoryService,
    MemoryTool,
    SecretScanner,
    memory_index,
    memory_pass,
    memory_tool,
    memory_tool_spec,
)
from src.agent.tools.planner import (
    PlannerTool,
    planner_tool,
    planner_tool_spec,
)
from src.agent.tools.registry import ToolRegistry, create_default_registry
from src.agent.tools.web import web_fetch_spec, web_search_spec

default_registry: ToolRegistry = create_default_registry(SETTINGS)

tools: List[Tool] = default_registry.get_specs()
functions: Dict[str, Callable] = default_registry.get_functions()

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "create_default_registry",
    "default_registry",
    "tools",
    "functions",
    "CalculatorTool",
    "calculator_tool",
    "calculator_tool_spec",
    "CacheTool",
    "cache_tool",
    "cache_tool_spec",
    "CliTool",
    "cli_tool",
    "cli_tool_spec",
    "CliPathGuard",
    "CliCommandValidator",
    "CliFileOperations",
    "safe_resolve_path",
    "_safe_path",
    "MemoryTool",
    "memory_tool",
    "memory_tool_spec",
    "MemoryFileStore",
    "MemoryService",
    "SecretScanner",
    "memory_index",
    "memory_pass",
    "PlannerTool",
    "planner_tool",
    "planner_tool_spec",
    "ToolBuilderTool",
    "tool_builder",
    "tool_builder_spec",
    "load_custom_tools",
    "web_search_spec",
    "web_fetch_spec",
]
