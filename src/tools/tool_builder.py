import json
import ast
import os
import re
import subprocess
import sys
from ollama import Tool
from pathlib import Path
from src.settings import AGENT_TOOLS_FOLDER_NAME

TOOLS_DIR = (Path.cwd() / AGENT_TOOLS_FOLDER_NAME).resolve()
NAME = re.compile(r"[a-z][a-z0-9_]{0,40}")
FILES = {"tool": "tool.py", "test": "test_tool.py"}

def _dir(name: str) -> Path:
    if not NAME.fullmatch(name):
        raise ValueError("Invalid tool name. Use lowercase letters, numbers and _")
    return TOOLS_DIR / name

def load_custom_tools():
    tools, funcs = [], {}
    for d in sorted(TOOLS_DIR.iterdir()) if TOOLS_DIR.exists() else []:
        spec_file = d / "spec.json"
        if d.is_dir() and spec_file.exists() and (d / ".passed").exists():
            props = json.loads(spec_file.read_text(encoding="utf-8"))["properties"]
            tools.append(
                Tool(
                    type="function",
                    function=Tool.Function(
                        name=d.name,
                        description=(d / "description.txt").read_text(encoding="utf-8").strip(),
                        parameters=Tool.Function.Parameters(
                            type="object",
                            properties=props,
                            required=list(props),
                        ),
                    ),
                )
            )
            funcs[d.name] = custom_tool_runner(d.name)
    return tools, funcs

def _exec(args: list[str], cwd: Path, timeout: int = 20):
    r = subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        env={"PATH": os.environ.get("PATH", "")},
    )
    return r.returncode, (r.stdout + r.stderr).strip()[:2000]


def custom_tool_runner(name: str):
    def run(**kwargs):
        return tool_builder(action="run", name=name, args=json.dumps(kwargs))
    return run


def tool_builder(
    action: str,
    name: str = "",
    description: str = "",
    code: str = "",
    test_code: str = "",
    args: str = "{}",
    file: str = "tool",
    start_line: int = 0,
    end_line: int = 0,
    params: str = "{}",
) -> str:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if action == "list":
            lines = []
            for d in sorted(TOOLS_DIR.iterdir()):
                if d.is_dir():
                    status = "tested" if (d / ".passed").exists() else "untested"
                    desc = (d / "description.txt").read_text(encoding="utf-8")
                    lines.append(f"{d.name} [{status}]: {desc}")
            return "\n".join(lines) or "No custom tools yet"

        if action != "list" and not name:
            return "Error: 'name' is required. Pass the same tool name in every call."

        d = _dir(name)

        if action == "create":
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return f"Error: syntax error in code at line {e.lineno}: {e.msg}. Call create again with the same name."
            try:
                ast.parse(test_code)
            except SyntaxError as e:
                return f"Error: syntax error in test_code at line {e.lineno}: {e.msg}. Call create again with the same name."
            if not any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
                return f"Error: code must define a function named exactly '{name}'. Call create again."

            header = f"from tool import {name}\n"
            if header not in test_code:
                test_code = header + test_code

            d.mkdir(exist_ok=True)
            (d / ".passed").unlink(missing_ok=True)
            (d / "tool.py").write_text(code, encoding="utf-8")
            (d / "test_tool.py").write_text(test_code, encoding="utf-8")
            (d / "description.txt").write_text(description, encoding="utf-8")
            return f"Saved {name}. Now call test with name='{name}'."

        if action == "read":
            lines = (d / FILES[file]).read_text(encoding="utf-8").splitlines()
            start = max(start_line, 1)
            end = min(end_line or len(lines), len(lines))
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

        if action == "update_lines":
            if file not in FILES:
                return "Error: file must be 'tool' or 'test'"
            p = d / FILES[file]
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            if start_line < 1 or end_line < start_line - 1 or start_line > len(lines) + 1:
                raise ValueError(f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines")
            new = code.splitlines(keepends=True)
            if new and not new[-1].endswith("\n"):
                new[-1] += "\n"
            lines[start_line - 1:end_line] = new
            p.write_text("".join(lines), encoding="utf-8")
            (d / ".passed").unlink(missing_ok=True)
            return f"Updated lines {start_line}-{end_line} of {FILES[file]}. Line numbers may have shifted. Call 'test' again."

        if action == "test":
            rc, out = _exec(["test_tool.py"], d)
            if rc == 0:
                (d / ".passed").touch()
                return f"PASSED\n{out}"
            (d / ".passed").unlink(missing_ok=True)
            return f"FAILED\n{out}"

        if action == "run":
            if not (d / ".passed").exists():
                return "Error: this tool has not passed its tests"
            json.loads(args)
            runner = f"import sys, json; from tool import {name} as f; print(f(**json.loads(sys.argv[1])))"
            return _exec(["-c", runner, args], d)[1]

        if action == "register":
            if not (d / ".passed").exists():
                return "Error: test the tool before registering"
            spec = json.loads(params)
            props = {
                k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                for k, v in spec.items()
            }
            (d / "spec.json").write_text(json.dumps({"properties": props}), encoding="utf-8")
            return f"Registered {name}. It is available right away."
        return f"Error: unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except Exception as e:
        return f"Error: {e}"