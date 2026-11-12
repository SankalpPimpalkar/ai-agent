from typing import Callable, Dict, List
from ollama import Tool
from src.settings import SETTINGS

from .calculator import calculator_tool, calculator_tool_spec
from .cli import cli_tool, cli_tool_spec
from .memory import memory_tool, memory_tool_spec
from .tool_builder import tool_builder, tool_builder_spec
from .web import web_search_spec, web_fetch_spec

tools: List[Tool] = []
functions: Dict[str, Callable] = {}

# Web tools
functions["web_search"] = SETTINGS.OLLAMA_CLIENT.web_search
functions["web_fetch"] = SETTINGS.OLLAMA_CLIENT.web_fetch

if SETTINGS.OLLAMA_WEB_SEARCH:
    tools.append(web_search_spec)
    tools.append(web_fetch_spec)

# Calculator tool
functions["calculator_tool"] = calculator_tool
tools.append(calculator_tool_spec)

# Tool builder
functions["tool_builder"] = tool_builder
tools.append(tool_builder_spec)

# Memory tool
functions["memory_tool"] = memory_tool
tools.append(memory_tool_spec)

# CLI tool
functions["cli_tool"] = cli_tool
tools.append(cli_tool_spec)

__all__ = [
    "tools",
    "functions",
    "calculator_tool",
    "calculator_tool_spec",
    "cli_tool",
    "cli_tool_spec",
    "memory_tool",
    "memory_tool_spec",
    "tool_builder",
    "tool_builder_spec",
    "web_search_spec",
    "web_fetch_spec",
]
