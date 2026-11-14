import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from ollama import Tool
from src.agent.tools.base import BaseTool
from src.agent.tools.builder.runner import ToolTestRunner
from src.agent.tools.builder.store import ToolFileStore
from src.agent.tools.builder.validator import ToolAstValidator
from src.agent.tools.cli.operations import read_numbered_lines, update_lines_in_text

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,40}")
MAX_CREATES = 3
MAX_TESTS = 3


class ToolBuilderTool(BaseTool):
    """
    Builds, tests, and registers dynamic custom Python tools at runtime.
    Actions: create_tool, test_tool, read_file, update_lines, delete_tool.
    """

    def __init__(
        self,
        validator: Optional[ToolAstValidator] = None,
        store: Optional[ToolFileStore] = None,
        runner: Optional[ToolTestRunner] = None,
    ):
        self.validator = validator or ToolAstValidator()
        self.store = store or ToolFileStore()
        self.runner = runner or ToolTestRunner()
        self._creates_count: int = 0
        self._tests_count: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "tool_builder"

    @property
    def description(self) -> str:
        return """
Build custom tools in Python when you need reusable functionality.
Actions:
- create_tool: name, code, test_code (defines and writes test)
- test_tool: name (runs unit tests; passes enable tool for subsequent steps)
- read_file: name, file_type ('tool' or 'test')
- update_lines: name, file_type, content, start_line, end_line
- delete_tool: name
""".strip()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_tool",
                        "test_tool",
                        "read_file",
                        "update_lines",
                        "delete_tool",
                    ],
                    "description": "Operation to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Snake_case tool name (letters, numbers, underscores)",
                },
                "code": {
                    "type": "string",
                    "description": "Python source code defining the function",
                },
                "test_code": {
                    "type": "string",
                    "description": "Python unittest test code verifying the tool",
                },
                "file_type": {
                    "type": "string",
                    "enum": ["tool", "test"],
                    "description": "File to read or update",
                },
                "content": {"type": "string", "description": "Replacement lines"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["action"],
        }

    def run(
        self,
        action: str,
        name: str = "",
        code: str = "",
        test_code: str = "",
        file_type: str = "tool",
        content: str = "",
        start_line: int = 1,
        end_line: int = 0,
        **kwargs: Any,
    ) -> str:
        try:
            action = (action or "").strip()
            name = (name or kwargs.get("tool_name", "")).strip()

            if not name:
                return "Error: 'name' is required"
            if not NAME_PATTERN.fullmatch(name):
                return "Error: tool name must be snake_case letters, digits, underscores"

            if action == "create_tool":
                return self.action_create(name, code, test_code)
            if action == "test_tool":
                return self.action_test(name)
            if action == "read_file":
                s = int(kwargs.get("start", start_line))
                e = int(kwargs.get("end", end_line))
                return self.action_read(name, file_type, s, e)
            if action == "update_lines":
                s = int(kwargs.get("start", start_line))
                e = int(kwargs.get("end", end_line))
                return self.action_update(name, file_type, content, s, e)
            if action == "delete_tool":
                return self.action_delete(name)

            return f"Error: unknown action '{action}'"
        except Exception as e:
            return f"Error: {e}"

    def action_create(self, name: str, code: str, test_code: str) -> str:
        if self._creates_count >= MAX_CREATES:
            return f"Error: create_tool limit ({MAX_CREATES}) reached"
        self._creates_count += 1

        props, required, description = self.validator.build_spec(code, name)
        self.store.write_file(name, "tool", code)
        self.store.write_file(name, "test", test_code)
        self.store.save_spec(name, props, required, description)
        return f"Tool '{name}' created. Run test_tool to verify and activate it."

    def action_test(self, name: str) -> str:
        runs = self._tests_count.get(name, 0)
        if runs >= MAX_TESTS:
            return f"Error: test_tool limit ({MAX_TESTS}) reached for tool '{name}'"
        self._tests_count[name] = runs + 1

        tool_dir = self.store.tool_dir(name)
        passed, output = self.runner.run_tests(tool_dir)
        if passed:
            self.store.mark_passed(name)
            return f"Tests passed! Tool '{name}' is verified and ready to use."
        return f"Tests failed:\n{output}"

    def action_read(
        self, name: str, file_type: str, start_line: int, end_line: int
    ) -> str:
        text = self.store.read_file(name, file_type)
        return read_numbered_lines(text, start_line, end_line)

    def action_update(
        self,
        name: str,
        file_type: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> str:
        orig = self.store.read_file(name, file_type)
        updated = update_lines_in_text(orig, content, start_line, end_line)
        self.store.write_file(name, file_type, updated)
        return f"Updated lines {start_line}-{end_line} in {file_type} file for tool '{name}'"

    def action_delete(self, name: str) -> str:
        d = self.store.root / name
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return f"Tool '{name}' deleted."
        return f"Tool '{name}' not found."


def load_custom_tools() -> Tuple[List[Tool], Dict[str, Callable]]:
    store = ToolFileStore()
    tools: List[Tool] = []
    funcs: Dict[str, Callable] = {}

    if not store.root.exists():
        return tools, funcs

    for d in store.root.iterdir():
        if not d.is_dir() or not (d / ".passed").exists():
            continue
        spec_file = d / "spec.json"
        tool_file = d / "tool.py"
        if not (spec_file.exists() and tool_file.exists()):
            continue
        try:
            raw_spec = json.loads(spec_file.read_text(encoding="utf-8"))
            tools.append(Tool(**raw_spec))
            mod_name = f"custom_tool_{d.name}"
            spec = importlib.util.spec_from_file_location(mod_name, tool_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, d.name):
                    funcs[d.name] = getattr(mod, d.name)
        except Exception:
            continue

    return tools, funcs


tool_builder_instance = ToolBuilderTool()


def tool_builder(
    action: str,
    name: str = "",
    code: str = "",
    test_code: str = "",
    file_type: str = "tool",
    content: str = "",
    start_line: int = 1,
    end_line: int = 0,
    **kwargs: Any,
) -> str:
    return tool_builder_instance.run(
        action=action,
        name=name,
        code=code,
        test_code=test_code,
        file_type=file_type,
        content=content,
        start_line=start_line,
        end_line=end_line,
        **kwargs,
    )


tool_builder_spec = tool_builder_instance.to_spec()
