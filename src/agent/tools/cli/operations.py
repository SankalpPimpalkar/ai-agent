import os
import shutil
from pathlib import Path
from typing import List, Union
from src.agent.tools.cli.path_guard import CliPathGuard
from src.agent.tools.cli.validator import clip_text
from src.agent.tools.state import STATE


def read_numbered_lines(
    lines_or_text: Union[List[str], str],
    start_line: int = 1,
    end_line: int = 0,
) -> str:
    if isinstance(lines_or_text, str):
        lines = lines_or_text.splitlines()
    else:
        lines = lines_or_text

    if not lines:
        return ""

    start = max(start_line, 1)
    end = end_line if end_line else len(lines)
    end = min(end, len(lines))

    if start > end:
        raise ValueError(f"Invalid range {start}-{end}, file has {len(lines)} lines")

    return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))


def update_lines_in_text(
    original_text: str,
    new_content: str,
    start_line: int,
    end_line: int,
) -> str:
    lines = original_text.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line - 1 or start_line > len(lines) + 1:
        raise ValueError(
            f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines"
        )
    before = lines[: start_line - 1]
    after = lines[end_line:] if end_line <= len(lines) else []
    replacement = new_content
    if replacement and not replacement.endswith("\n") and after:
        replacement += "\n"
    return "".join(before) + replacement + "".join(after)


class CliFileOperations:
    """Handles file and directory manipulation within a project artifact."""

    def __init__(self, path_guard: CliPathGuard):
        self.path_guard = path_guard

    @staticmethod
    def sanitize(content: str, filename: str = "") -> str:
        if not (filename.endswith(".py") or "import " in content or "def " in content):
            return content
        clean = content.replace("from tools import ", "from agent_tools import ")
        clean = clean.replace("import tools.", "import agent_tools.")
        return clean

    def create_file(self, project: str, path: str, content: str) -> str:
        target = self.path_guard.safe_path(path, project)
        target.parent.mkdir(parents=True, exist_ok=True)
        sanitized = self.sanitize(content, target.name)
        target.write_text(sanitized, encoding="utf-8")
        STATE.register_file(path)
        return f"Created {path}"

    def update_file(self, project: str, path: str, content: str) -> str:
        target = self.path_guard.safe_path(path, project)
        if not target.exists():
            return f"Error: file not found: {path}"
        sanitized = self.sanitize(content, target.name)
        target.write_text(sanitized, encoding="utf-8")
        STATE.register_file(path)
        return f"Updated {path}"

    def read_file(
        self, project: str, path: str, start_line: int = 1, end_line: int = 0
    ) -> str:
        target = self.path_guard.safe_path(path, project)
        if not target.exists():
            return f"Error: file not found: {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return read_numbered_lines(text, start_line=start_line, end_line=end_line)

    def update_lines(
        self,
        project: str,
        path: str,
        content: str,
        start_line: int,
        end_line: int,
    ) -> str:
        target = self.path_guard.safe_path(path, project)
        if not target.exists():
            return f"Error: file not found: {path}"
        orig = target.read_text(encoding="utf-8", errors="replace")
        updated = update_lines_in_text(orig, content, start_line, end_line)
        target.write_text(self.sanitize(updated, target.name), encoding="utf-8")
        return f"Updated lines {start_line}-{end_line} in {path}"

    def delete_file(self, project: str, path: str) -> str:
        target = self.path_guard.safe_path(path, project)
        if not target.exists():
            return f"Error: file not found: {path}"
        target.unlink()
        STATE.unregister_file(path)
        return f"Deleted {path}"

    def list_files(self, project: str, path: str = ".") -> str:
        target = self.path_guard.safe_path(path, project)
        if not target.exists():
            return f"Error: path not found: {path}"
        if not target.is_dir():
            return f"Error: not a directory: {path}"
        items = []
        for entry in sorted(target.iterdir()):
            if entry.name.startswith("."):
                continue
            suffix = "/" if entry.is_dir() else ""
            items.append(f"{entry.name}{suffix}")
        return "\n".join(items) if items else "Directory is empty"
