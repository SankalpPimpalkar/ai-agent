import json
import ast
import os
import re
import shutil
import subprocess
import sys
from ollama import Tool
from pathlib import Path
from src.settings import SETTINGS
from src.tools.state import STATE

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


def load_custom_tools():
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
    return r.returncode, (r.stdout + r.stderr).strip()[-2000:]


def custom_tool_runner(name: str):
    def run(**kwargs):
        return tool_builder(action="run", name=name, run_args=json.dumps(kwargs))

    return run


def _build_spec(code: str, name: str):
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
    name = name or tool_name
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if action == "list":
            STATE.tools_listed = True
            lines = []
            for d in sorted(TOOLS_DIR.iterdir()):
                if d.is_dir():
                    status = "tested" if (d / ".passed").exists() else "untested"
                    desc_file = d / "description.txt"
                    desc = (
                        desc_file.read_text(encoding="utf-8")
                        if desc_file.exists()
                        else ""
                    )
                    lines.append(f"{d.name} [{status}]: {desc}")
            return "\n".join(lines) or "No custom tools yet"

        if not name:
            return "Error: 'name' is required. Pass the same tool name in every call."

        d = _dir(name)

        if action == "create":
            if not STATE.tools_listed:
                return "Error: call tool_builder with action='list' first."
            if STATE.creates >= MAX_CREATES:
                return (
                    "Error: create limit reached. Stop building tools. "
                    "Answer with what you have and say what failed."
                )
            if not code.strip():
                return "Error: 'code' is empty. Pass the Python code as the top-level 'code' argument, next to 'action' and 'name'."
            if not test_code.strip():
                return "Error: 'test_code' is empty. Pass the assert statements as the top-level 'test_code' argument."

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

            STATE.creates += 1

            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            (d / "tool.py").write_text(code, encoding="utf-8")
            (d / "test_tool.py").write_text(test_code, encoding="utf-8")
            (d / "description.txt").write_text(description, encoding="utf-8")
            _write_spec(d, props, required)
            return f"Saved {name}. Now call test with name='{name}'."

        if action in ("read", "update_lines") and file not in FILES:
            return "Error: file must be 'tool' or 'test'"

        if action == "delete":
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                return f"Deleted tool '{name}'. You can now create it again from scratch with action='create'."
            return f"Tool '{name}' does not exist."

        if not d.exists():
            return f"Error: tool '{name}' does not exist. Call list to see existing tools."

        if action == "read":
            lines = (d / FILES[file]).read_text(encoding="utf-8").splitlines()
            start = max(start_line, 1)
            end = end_line if end_line > 0 else len(lines)
            end = min(end, len(lines))
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

        if action == "update_lines":
            if end_line < 0:
                return "Error: end_line is required. To rewrite the whole file, call create instead."

            p = d / FILES[file]
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            if (
                start_line < 1
                or end_line < start_line - 1
                or start_line > len(lines) + 1
            ):
                raise ValueError(
                    f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines"
                )
            new = code.splitlines(keepends=True)
            if new and not new[-1].endswith("\n"):
                new[-1] += "\n"
            lines[start_line - 1 : end_line] = new
            p.write_text("".join(lines), encoding="utf-8")
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

        if action == "test":
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

        if action == "run":
            try:
                json.loads(run_args)
            except json.JSONDecodeError:
                return "Error: run_args must be a valid JSON object string, e.g. '{\"n\": 5}'"
            runner = "import sys, json; from tool import %s as f; print(f(**json.loads(sys.argv[1])))" % name
            out = _exec(["-c", runner, run_args], d)[1]
            return out if (d / ".passed").exists() else f"[UNTESTED TOOL] {out}"

        if action == "register":
            return "No need to register. The tool is available as soon as its test passes."

        return f"Error: unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: timed out"
    except Exception as e:
        return f"Error: {e}"