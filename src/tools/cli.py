import os
import re
import shutil
import subprocess
from pathlib import Path
from ollama import Tool
from src.settings import SETTINGS
from src.utils.files import clip_text, safe_resolve_path, read_numbered_lines, update_lines_in_text

ARTIFACTS_FOLDER_NAME = SETTINGS.ARTIFACTS_FOLDER_NAME
ROOT = (Path.cwd() / ARTIFACTS_FOLDER_NAME).resolve()
DEV_SERVERS = ["npm run dev", "npm start", "yarn dev", "pnpm dev"]
PROJECT_NAME = re.compile(r"[A-Za-z0-9_-]+")


def _project_root(project: str) -> Path:
    if not PROJECT_NAME.fullmatch(project):
        raise ValueError("Invalid project name. Use letters, numbers, - and _ only")
    p = (ROOT / project).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_path(p: str, project: str) -> Path:
    base = _project_root(project)
    clean = p.strip() if p else "."
    try:
        return safe_resolve_path(base, clean)
    except ValueError:
        raise ValueError(f"Path escapes project: {p}")


def list_projects_action() -> str:
    ROOT.mkdir(parents=True, exist_ok=True)
    names = sorted(e.name for e in ROOT.iterdir() if e.is_dir())
    return "\n".join(names) or "No projects yet"


def run_action(project: str, path: str, command: str) -> str:
    cwd = _safe_path(path, project)
    cwd.mkdir(parents=True, exist_ok=True)

    cmd_strip = command.strip()
    if not cmd_strip:
        raise ValueError("Empty command")

    cmd_lower = cmd_strip.lower()
    for ds in DEV_SERVERS:
        if cmd_lower.startswith(ds):
            raise ValueError("Dev servers never exit. Use 'npm run build' to verify")

    env = {
        **os.environ,
        "CI": "1",
        "PYTHONPATH": f"{cwd}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
    }

    result = subprocess.run(
        cmd_strip,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        shell=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    if result.returncode != 0:
        return f"Error: {clip_text(result.stderr or result.stdout)}"
    return clip_text(result.stdout) or "Command completed"


def create_file_action(project: str, path: str, content: str) -> str:
    p = _safe_path(path, project)
    if p.is_dir():
        default_file = "app.py" if ("def " in content or "import " in content) else "output.txt"
        p = p / default_file
        path = f"{path.rstrip('/')}/{default_file}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Created {path}"


def read_file_action(project: str, path: str, start_line: int, end_line: int) -> str:
    p = _safe_path(path, project)
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{path}' in project '{project}'")
    lines = p.read_text(encoding="utf-8").splitlines()
    formatted = read_numbered_lines(lines, start_line, end_line)
    return clip_text(formatted, 4000)


def update_file_action(project: str, path: str, content: str, append: bool) -> str:
    p = _safe_path(path, project)
    if p.is_dir():
        default_file = "app.py" if ("def " in content or "import " in content) else "output.txt"
        p = p / default_file
        path = f"{path.rstrip('/')}/{default_file}"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", encoding="utf-8") as f:
        f.write(content)
    return f"Updated {path}"


def update_lines_action(
    project: str, path: str, content: str, start_line: int, end_line: int
) -> str:
    p = _safe_path(path, project)
    if not p.exists():
        raise FileNotFoundError(f"File not found: '{path}' in project '{project}'")
    original = p.read_text(encoding="utf-8")
    updated = update_lines_in_text(original, content, start_line, end_line)
    p.write_text(updated, encoding="utf-8")
    return f"Updated lines {start_line}-{end_line} in {path}. Line numbers below this point may have shifted."


def delete_file_action(project: str, path: str) -> str:
    p = _safe_path(path, project)
    if not p.exists():
        return f"File '{path}' does not exist"
    p.unlink()
    return f"Deleted {path}"


def create_folder_action(project: str, path: str) -> str:
    clean_path = path.strip().removeprefix("./").removesuffix("/") if path else "."
    if clean_path in (".", "", project):
        _project_root(project)
        return f"Created folder {project}"
    p = _safe_path(clean_path, project)
    p.mkdir(parents=True, exist_ok=True)
    return f"Created folder {clean_path}"


def delete_folder_action(project: str, path: str) -> str:
    clean_path = path.strip().removeprefix("./").removesuffix("/") if path else "."
    if clean_path in (".", ""):
        raise ValueError("Cannot delete the project root")
    p = _safe_path(clean_path, project)
    base = _project_root(project)
    if p == base:
        raise ValueError("Cannot delete the project root")
    if not p.exists():
        return f"Folder '{clean_path}' does not exist"
    shutil.rmtree(p)
    return f"Deleted folder {clean_path}"


def list_files_action(project: str, path: str) -> str:
    p = _safe_path(path, project)
    if not p.exists():
        return "Folder is empty"
    entries = sorted(p.iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "Folder is empty"


def cli_tool(
    action: str,
    project: str = "",
    path: str = ".",
    content: str = "",
    command: str = "",
    append: bool = False,
    start_line: int = 0,
    end_line: int = 0,
    **kwargs,
) -> str:
    ROOT.mkdir(parents=True, exist_ok=True)

    # Normalize alias arguments from LLM
    project = (
        project
        or kwargs.get("project_name", "")
        or kwargs.get("folder", "")
        or kwargs.get("dir", "")
        or kwargs.get("app", "")
    )
    content = (
        content
        or kwargs.get("code", "")
        or kwargs.get("text", "")
        or kwargs.get("data", "")
    )
    command = command or kwargs.get("cmd", "") or kwargs.get("script", "")
    if path in (".", ""):
        path = kwargs.get("file", kwargs.get("filename", kwargs.get("filepath", ".")))
    if not project and action in (
        "run",
        "create_file",
        "read_file",
        "update_file",
        "update_lines",
        "list",
    ):
        project = "app"

    try:
        if action == "list_projects":
            return list_projects_action()

        if not project:
            return "Error: 'project' is required. Call list_projects to see existing ones"

        if action == "run":
            return run_action(project, path, command)
        elif action == "create_file":
            return create_file_action(project, path, content)
        elif action == "read_file":
            return read_file_action(project, path, start_line, end_line)
        elif action == "update_file":
            return update_file_action(project, path, content, append)
        elif action == "update_lines":
            return update_lines_action(project, path, content, start_line, end_line)
        elif action == "delete_file":
            return delete_file_action(project, path)
        elif action == "create_folder":
            return create_folder_action(project, path)
        elif action == "delete_folder":
            return delete_folder_action(project, path)
        elif action == "list":
            return list_files_action(project, path)

        return f"Error: Unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"


cli_tool_spec = Tool(
    type="function",
    function=Tool.Function(
        name="cli_tool",
        description=f"""
Manage and run project artifacts in {ARTIFACTS_FOLDER_NAME}/.
Use this tool ONLY when the user explicitly asks to build a project, scaffold an application, or work with project files on disk.
Each project lives in its own dedicated folder inside {ARTIFACTS_FOLDER_NAME}/ (passed as 'project', e.g. 'project="markdown-converter"').
Artifacts must persist permanently — NEVER delete project files or folders when done.
To delete records or items within an application (e.g. deleting a task in a todo app), run the app's CLI command using action='run' (e.g. 'python3 todo.py delete ...'), DO NOT use action='delete_file'.

Actions:
- 'create_file': Create a new file within the project with 'content' and 'path' (e.g. 'app.py').
- 'run': Execute a shell command inside the project directory (e.g. 'python3 app.py').
- 'read_file': Read a file with numbered lines, optionally specifying start_line and end_line.
- 'update_file': Overwrite or append to an entire file.
- 'update_lines': Replace a specific line range in a file with new content.
- 'delete_file': Delete an unwanted source code file in the project. Never use this to delete user data or completed artifacts.
- 'create_folder': Create a subdirectory inside the project.
- 'delete_folder': Delete a subdirectory inside the project.
- 'list_projects': List existing project directories.
- 'list': List files in the project.
""".strip(),
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "action": {
                    "type": "string",
                    "enum": [
                        "run",
                        "create_file",
                        "read_file",
                        "update_file",
                        "update_lines",
                        "delete_file",
                        "create_folder",
                        "delete_folder",
                        "list_projects",
                        "list",
                    ],
                    "description": "Operation to perform.",
                },
                "project": {
                    "type": "string",
                    "description": f"Project folder name inside {ARTIFACTS_FOLDER_NAME}/ (e.g. 'todo-app', 'markdown-converter').",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file or directory path within the project folder. Defaults to '.'.",
                },
                "content": {
                    "type": "string",
                    "description": "Content for create_file, update_file, or update_lines.",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute when action='run' (e.g. 'python3 app.py').",
                },
                "append": {
                    "type": "boolean",
                    "description": "If true, update_file appends content rather than overwriting.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based starting line number for read_file or update_lines.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based ending line number (inclusive) for read_file or update_lines.",
                },
            },
            required=["action"],
        ),
    ),
)
