import os
import re
import shutil
import subprocess
from pathlib import Path
from src.settings import SETTINGS

BLACK_LISTED = {
    "rm",
    "rmdir",
    "del",
    "format",
    "mkfs",
    "dd",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "sudo",
    "su",
    "chmod",
    "chown",
    "kill",
    "killall",
    "curl",
    "wget",
    "nc",
    "ssh",
    "scp",
    "powershell",
    "cmd",
    "bash",
    "sh",
    "zsh",
    "eval",
    "exec",
}

ARTIFACTS_FOLDER_NAME = SETTINGS.ARTIFACTS_FOLDER_NAME
ROOT = (Path.cwd() / ARTIFACTS_FOLDER_NAME).resolve()
ALLOWED_BINARIES = {"npm", "npx", "pnpm", "yarn", "git", "node", "pip", "python"}
SHELL_META = re.compile(r"[;&|`$<>\n\r]")
PROJECT_NAME = re.compile(r"[A-Za-z0-9_-]+")

def _project_root(project: str) -> Path:
    if not PROJECT_NAME.fullmatch(project):
        raise ValueError("Invalid project name. Use letters, numbers, - and _ only")
    return (ROOT / project).resolve()

def _safe_path(p: str, project: str) -> Path:
    base = _project_root(project)
    full = (base / p).resolve()
    if full != base and base not in full.parents:
        raise ValueError(f"Path escapes project: {p}")
    return full

def _validate_command(command: str) -> list[str]:
    if SHELL_META.search(command):
        raise ValueError("Shell operators are not allowed")

    parts = command.strip().split()
    if not parts:
        raise ValueError("Empty command")

    name = Path(parts[0]).name.lower()
    if name in BLACK_LISTED:
        raise ValueError(f"Blacklisted command: {name}")
    if name not in ALLOWED_BINARIES:
        raise ValueError(f"Command not allowed: {name}")

    for arg in parts[1:]:
        if arg.lower() in BLACK_LISTED:
            raise ValueError(f"Blacklisted argument: {arg}")
        if Path(arg).is_absolute() or ".." in re.split(r"[\\/]", arg):
            raise ValueError(f"Unsafe path argument: {arg}")
    return parts


def cli_tool(
    action: str,
    project: str = "",
    path: str = ".",
    content: str = "",
    command: str = "",
    append: bool = False,
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    ROOT.mkdir(parents=True, exist_ok=True)

    try:
        if action == "list_projects":
            names = sorted(e.name for e in ROOT.iterdir() if e.is_dir())
            return "\n".join(names) or "No projects yet"

        if not project:
            return "Error: 'project' is required. Call list_projects to see existing ones"

        base = _project_root(project)

        if action == "run":
            cwd = _safe_path(path, project)
            cwd.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                _validate_command(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,
                shell=False,
                stdin=subprocess.DEVNULL,   
                env={**os.environ, "CI": "1"},
            )
            if result.returncode != 0:
                return f"Error: {result.stderr or result.stdout}"
            return result.stdout or "Command completed"

        if action == "create_file":
            p = _safe_path(path, project)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Created {path}"

        if action == "read_file":
            lines = _safe_path(path, project).read_text(encoding="utf-8").splitlines()
            start = max(start_line, 1)
            end = end_line if end_line else len(lines)
            return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))

        if action == "update_file":
            p = _safe_path(path, project)
            with open(p, "a" if append else "w", encoding="utf-8") as f:
                f.write(content)
            return f"Updated {path}"

        if action == "update_lines":
            p = _safe_path(path, project)
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            if start_line < 1 or end_line < start_line - 1 or start_line > len(lines) + 1:
                raise ValueError(f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines")
            new = content.splitlines(keepends=True)
            if new and not new[-1].endswith("\n"):
                new[-1] += "\n"
            lines[start_line - 1:end_line] = new
            p.write_text("".join(lines), encoding="utf-8")
            return f"Updated lines {start_line}-{end_line} in {path}. Line numbers below this point may have shifted."

        if action == "delete_file":
            _safe_path(path, project).unlink()
            return f"Deleted {path}"

        if action == "create_folder":
            _safe_path(path, project).mkdir(parents=True, exist_ok=True)
            return f"Created folder {path}"

        if action == "delete_folder":
            p = _safe_path(path, project)
            if p == base:
                raise ValueError("Cannot delete the project root")
            shutil.rmtree(p)
            return f"Deleted folder {path}"

        if action == "list":
            entries = sorted(_safe_path(path, project).iterdir())
            return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)

        return f"Error: Unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"
