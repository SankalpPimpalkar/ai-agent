import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from ollama import Tool
from src.settings import SETTINGS
from src.tools.state import STATE
from src.utils import read_numbered_lines, update_lines_in_text, run_subprocess

AGENT_TOOLS_FOLDER_NAME = SETTINGS.AGENT_TOOLS_FOLDER_NAME
TOOLS_DIR = (Path.cwd() / AGENT_TOOLS_FOLDER_NAME).resolve()
NAME = re.compile(r"[a-z][a-z0-9_]{0,40}")
FILES = {"tool": "tool.py", "test": "test_tool.py"}
MAX_CREATES = 3
MAX_TESTS = 3


def _json_type(annotation: str) -> str:
    base = annotation.split("[", 1)[0].strip().lower()
    return {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }.get(base, "string")


def _dir(name: str) -> Path:
    if not NAME.fullmatch(name):
        raise ValueError("Invalid tool name. Use lowercase letters, numbers and _")
    return TOOLS_DIR / name


def _exec(args: List[str], cwd: Path, timeout: int = 20) -> Tuple[int, str]:
    pythonpath = f"{TOOLS_DIR}:{cwd}:{os.environ.get('PYTHONPATH', '')}".rstrip(":")
    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
    }
    return run_subprocess(
        [sys.executable, "-B", *args],
        cwd=cwd,
        timeout=timeout,
        env=env,
        max_output_chars=2000,
    )


def _inject_custom_tool_imports(code: str, current_tool_name: str) -> str:
    """
    If code references any existing passed custom tools from TOOLS_DIR
    without defining or importing them, automatically add imports at the top.
    """
    if not TOOLS_DIR.exists():
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    defined_names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined_names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined_names.add(alias.asname or alias.name)

    imports_to_add = []
    for d in sorted(TOOLS_DIR.iterdir()):
        if d.is_dir() and d.name != current_tool_name and (d / ".passed").exists():
            tool_name = d.name
            if tool_name not in defined_names:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == tool_name:
                        imports_to_add.append(f"from {tool_name} import {tool_name}\n")
                        break

    if imports_to_add:
        return "".join(imports_to_add) + code
    return code


def custom_tool_runner(name: str):
    def run(**kwargs):
        return tool_builder(action="run", name=name, run_args=json.dumps(kwargs))

    return run


def load_custom_tools() -> Tuple[List[Tool], Dict[str, callable]]:
    tools, funcs = [], {}
    for d in sorted(TOOLS_DIR.iterdir()) if TOOLS_DIR.exists() else []:
        spec_file = d / "spec.json"
        if d.is_dir() and spec_file.exists() and (d / ".passed").exists():
            spec = json.loads(spec_file.read_text(encoding="utf-8"))
            tools.append(
                Tool(
                    type="function",
                    function=Tool.Function(
                        name=d.name,
                        description=(d / "description.txt")
                        .read_text(encoding="utf-8")
                        .strip(),
                        parameters=Tool.Function.Parameters(
                            type="object",
                            properties=spec["properties"],
                            required=spec.get("required", list(spec["properties"])),
                        ),
                    ),
                )
            )
            funcs[d.name] = custom_tool_runner(d.name)
    return tools, funcs


def _build_spec(code: str, name: str) -> Tuple[dict, list, str]:
    tree = ast.parse(code)
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    if fn is None:
        raise ValueError(f"code must define a function named exactly '{name}'")
    args = fn.args.args
    first_default = len(args) - len(fn.args.defaults)
    props, required = {}, []
    for i, a in enumerate(args):
        t = ast.unparse(a.annotation) if a.annotation else "str"
        props[a.arg] = {
            "type": _json_type(t),
            "description": a.arg.replace("_", " "),
        }
        if i < first_default:
            required.append(a.arg)
    doc = (ast.get_docstring(fn) or "").strip().splitlines()
    return props, required, (doc[0] if doc else "")


def _write_spec(d: Path, props: dict, required: list):
    (d / "spec.json").write_text(
        json.dumps({"properties": props, "required": required}), encoding="utf-8"
    )


def list_tools_action() -> str:
    STATE.tools_listed = True
    lines = []
    for d in sorted(TOOLS_DIR.iterdir()):
        if d.is_dir():
            status = "tested" if (d / ".passed").exists() else "untested"
            desc_file = d / "description.txt"
            desc = desc_file.read_text(encoding="utf-8") if desc_file.exists() else ""
            lines.append(f"{d.name} [{status}]: {desc}")
    return "\n".join(lines) or "No custom tools yet"


def create_tool_action(
    d: Path, name: str, description: str, code: str, test_code: str
) -> str:
    STATE.tools_listed = True
    if STATE.creates >= MAX_CREATES:
        return (
            "Error: create limit reached. Stop building tools. "
            "Answer with what you have and say what failed."
        )
    if not code.strip():
        return "Error: 'code' is empty. Pass the Python code as the top-level 'code' argument, next to 'action' and 'name'."
    if not test_code.strip():
        return "Error: 'test_code' is empty. Pass the assert statements as the top-level 'test_code' argument."

    # Automatically resolve cross-tool imports if referencing another passed custom tool
    code = _inject_custom_tool_imports(code, current_tool_name=name)

    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Error: syntax error in code at line {e.lineno}: {e.msg}. Call create again with the same name."
    try:
        ast.parse(test_code)
    except SyntaxError as e:
        return f"Error: syntax error in test_code at line {e.lineno}: {e.msg}. Call create again with the same name."
    try:
        props, required, doc_line = _build_spec(code, name)
    except ValueError as e:
        return f"Error: {e}. Call create again."

    description = description.strip() or doc_line
    if not description:
        return "Error: add a docstring to the function or pass 'description'. Call create again with the same name."

    header = f"from tool import {name}\n"
    if header not in test_code:
        test_code = header + test_code
    test_code = _inject_custom_tool_imports(test_code, current_tool_name=name)

    STATE.creates += 1

    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    (d / "tool.py").write_text(code, encoding="utf-8")
    (d / "test_tool.py").write_text(test_code, encoding="utf-8")
    (d / "description.txt").write_text(description, encoding="utf-8")
    (d / "__init__.py").write_text("from .tool import *\n", encoding="utf-8")
    _write_spec(d, props, required)
    return f"Saved {name}. Now call test with name='{name}'."


def read_tool_action(d: Path, file: str, start_line: int, end_line: int) -> str:
    lines = (d / FILES[file]).read_text(encoding="utf-8").splitlines()
    return read_numbered_lines(lines, start_line, end_line if end_line > 0 else 0)


def update_lines_tool_action(
    d: Path, name: str, file: str, code: str, start_line: int, end_line: int
) -> str:
    if end_line < 0:
        return "Error: end_line is required. To rewrite the whole file, call create instead."

    p = d / FILES[file]
    original = p.read_text(encoding="utf-8")
    updated = update_lines_in_text(original, code, start_line, end_line)
    if file == "tool":
        updated = _inject_custom_tool_imports(updated, current_tool_name=name)
    p.write_text(updated, encoding="utf-8")
    (d / ".passed").unlink(missing_ok=True)

    if file == "tool":
        try:
            props, required, doc_line = _build_spec(
                p.read_text(encoding="utf-8"), name
            )
            _write_spec(d, props, required)
            if doc_line:
                (d / "description.txt").write_text(doc_line, encoding="utf-8")
        except Exception as e:
            return f"Error: updated, but the code no longer parses: {e}. Fix it with update_lines."

    return f"Updated lines {start_line}-{end_line} of {FILES[file]}. Line numbers may have shifted. Call 'test' again."


def test_tool_action(d: Path, name: str) -> str:
    attempts_file = d / ".test_attempts"
    attempts = 0
    if attempts_file.exists():
        try:
            attempts = int(attempts_file.read_text(encoding="utf-8").strip())
        except Exception:
            attempts = 0
    attempts += 1
    attempts_file.write_text(str(attempts), encoding="utf-8")

    rc, out = _exec(["test_tool.py"], d)
    if rc == 0:
        (d / ".passed").touch()
        attempts_file.unlink(missing_ok=True)
        return f"PASSED. The tool '{name}' is now available.\n{out}"

    (d / ".passed").unlink(missing_ok=True)
    if attempts >= MAX_TESTS:
        shutil.rmtree(d, ignore_errors=True)
        if STATE.creates >= MAX_CREATES:
            return (
                f"FAILED (attempt {MAX_TESTS}/{MAX_TESTS}).\n{out}\n"
                f"Tool '{name}' was deleted and the create limit is reached. "
                f"Stop building. Answer with what you have and say what failed."
            )
        return (
            f"FAILED (attempt {MAX_TESTS}/{MAX_TESTS}).\n{out}\n"
            f"Tool '{name}' failed {MAX_TESTS} test attempts and has been deleted. "
            f"Do not call update_lines. Call tool_builder with action='create' to create the tool again from scratch with fixed code and tests."
        )

    remaining = MAX_TESTS - attempts
    return (
        f"FAILED (attempt {attempts}/{MAX_TESTS}).\n{out}\n"
        f"You have {remaining} test retry(ies) remaining before '{name}' is deleted. "
        f"Fix the failing test/code with update_lines, then test again."
    )


def run_tool_action(d: Path, name: str, run_args: str) -> str:
    if isinstance(run_args, dict):
        run_args = json.dumps(run_args)
    elif not isinstance(run_args, str):
        run_args = str(run_args)

    if not (d / ".passed").exists():
        return f"Error: Tool '{name}' has not passed tests yet. Call tool_builder with action='test' and make it pass first."

    try:
        json.loads(run_args)
    except json.JSONDecodeError:
        return "Error: run_args must be a valid JSON object string, e.g. '{\"n\": 5}'"
    runner = "import sys, json; from tool import %s as f; print(f(**json.loads(sys.argv[1])))" % name
    out = _exec(["-c", runner, run_args], d)[1]
    return out


def delete_tool_action(d: Path, name: str) -> str:
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return f"Deleted tool '{name}'. You can now create it again from scratch with action='create'."
    return f"Tool '{name}' does not exist."


def tool_builder(
    action: str,
    name: str = "",
    tool_name: str = "",
    description: str = "",
    code: str = "",
    test_code: str = "",
    run_args: str = "{}",
    file: str = "tool",
    start_line: int = 0,
    end_line: int = -1,
    params: str = "{}",  # ignored, kept so old calls do not break
    **kwargs,
) -> str:
    name = name or tool_name or kwargs.get("name", "") or kwargs.get("tool_name", "")
    code = code or kwargs.get("content", "")
    test_code = test_code or kwargs.get("test", "") or kwargs.get("tests", "")

    if isinstance(run_args, dict):
        run_args = json.dumps(run_args)

    # Normalize file argument if provided with extension
    if file in ("tool.py", "tool"):
        file = "tool"
    elif file in ("test_tool.py", "test.py", "test"):
        file = "test"

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if action == "list":
            return list_tools_action()

        if not name:
            return "Error: 'name' is required. Pass the same tool name in every call."

        d = _dir(name)

        if action == "create":
            return create_tool_action(d, name, description, code, test_code)

        if action in ("read", "update_lines") and file not in FILES:
            return "Error: file must be 'tool' or 'test'"

        if action == "delete":
            return delete_tool_action(d, name)

        if not d.exists():
            return f"Error: tool '{name}' does not exist. Call list to see existing tools."

        if action == "read":
            return read_tool_action(d, file, start_line, end_line)
        elif action == "update_lines":
            return update_lines_tool_action(d, name, file, code, start_line, end_line)
        elif action == "test":
            return test_tool_action(d, name)
        elif action == "run":
            return run_tool_action(d, name, run_args)
        elif action == "register":
            return "No need to register. The tool is available as soon as its test passes."

        return f"Error: unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except Exception as e:
        return f"Error: {e}"


tool_builder_spec = Tool(
    type="function",
    function=Tool.Function(
        name="tool_builder",
        description="""
Build, test, and manage custom reusable Python tools to solve problems that cannot be solved with built-in tools.
Use this ONLY when existing built-in tools (calculator, cli_tool, memory_tool, web) cannot solve the problem.

Workflow:
1. Call action='list' to check if a relevant custom tool already exists.
2. Call action='create' with name, code, and test_code.
3. Call action='test' with name. Tests must pass before the tool can be run. If testing fails, use action='update_lines' to fix code or tests, then test again.
4. Once tests pass, the tool becomes available as a first-class tool or via action='run'.

Actions:
- 'list': List all custom tools and their test status.
- 'create': Define a new tool. 'code' must define a function named exactly like 'name' with type hints and a docstring. 'test_code' must contain assertions.
- 'test': Run the test suite for the tool.
- 'run': Execute a passed tool with 'run_args' as a JSON string (e.g. '{"n": 27}').
- 'read': View lines from a tool's 'tool' or 'test' file.
- 'update_lines': Modify lines in a tool's code or test file.
- 'delete': Delete a custom tool.
""".strip(),
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "create",
                        "test",
                        "run",
                        "read",
                        "update_lines",
                        "delete",
                    ],
                    "description": "Operation to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Tool name in snake_case. Required for every action except 'list'.",
                },
                "code": {
                    "type": "string",
                    "description": "Python code for 'create', or replacement text for 'update_lines'.",
                },
                "test_code": {
                    "type": "string",
                    "description": "Python test code using assert statements. Required for 'create'.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional tool description. Defaults to the first line of the function docstring.",
                },
                "run_args": {
                    "type": "string",
                    "description": "JSON object with the tool's arguments when action='run', e.g. '{\"n\": 27}'.",
                },
                "file": {
                    "type": "string",
                    "enum": ["tool", "test"],
                    "description": "Which file to read or edit for 'read' and 'update_lines'.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line (1-indexed) for 'read' and 'update_lines'.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line (inclusive) for 'read' and 'update_lines'.",
                },
            },
            required=["action"],
        ),
    ),
)
