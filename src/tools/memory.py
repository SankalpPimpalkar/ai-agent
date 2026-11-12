import re
import threading
from pathlib import Path
from typing import List
from ollama import Tool
from src.settings import BASE_DIR, SETTINGS
from src.tools.state import STATE
from src.utils.files import read_numbered_lines

MEMORY_DIR = BASE_DIR / SETTINGS.MEMORY_FOLDER_NAME
SECRET = re.compile(r"(api[_-]?key|password|passwd|secret|token)\s*[:=]|sk-[A-Za-z0-9]{16,}", re.I)
WRITES = {"create", "append", "update_lines", "delete_lines", "delete_file"}
MAX_FILES = 50
MAX_LINES = 200
_LOCK = threading.Lock()
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def _path(name: str) -> Path:
    name = name.strip().lower().removesuffix(".md")
    if not _NAME.match(name):
        raise ValueError("Invalid file name. Use lowercase letters, numbers, '-' or '_'")
    return MEMORY_DIR / f"{name}.md"


def _load(p: Path) -> List[str]:
    if not p.exists():
        raise ValueError(f"'{p.stem}' does not exist. Call list or create it first")
    return p.read_text(encoding="utf-8").splitlines()


def _save(p: Path, lines: List[str]):
    if len(lines) > MAX_LINES:
        raise ValueError(f"File is full ({MAX_LINES} lines max). Merge or delete lines first")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _check_range(start_line: int, end_line: int, lines: List[str]):
    if not 1 <= start_line <= end_line <= len(lines):
        raise ValueError(
            f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines"
        )


def memory_index() -> str:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MEMORY_DIR.glob("*.md"))
    if not files:
        return "No memory files yet"
    out = []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        out.append(f"{f.stem} ({len(lines)} lines): {lines[0] if lines else '(empty)'}")
    return "\n".join(out)


def create_memory_action(p: Path, content: str) -> str:
    if p.exists():
        raise ValueError(f"'{p.stem}' already exists. Use append or update_lines")
    if len(list(MEMORY_DIR.glob("*.md"))) >= MAX_FILES:
        raise ValueError(f"Too many files ({MAX_FILES} max). Merge or delete some")
    lines = content.splitlines()
    if not lines:
        raise ValueError("'content' is required. Line 1 should describe the file")
    _save(p, lines)
    return f"Created {p.stem}"


def read_memory_action(p: Path, lines: List[str], start_line: int, end_line: int) -> str:
    if not lines:
        return f"{p.stem} is empty"
    return read_numbered_lines(lines, start_line, end_line)


def append_memory_action(p: Path, lines: List[str], content: str) -> str:
    new = [line.strip() for line in content.splitlines() if line.strip()]
    if not new:
        raise ValueError("'content' is required")
    existing_lower = [l.strip().lower() for l in lines]
    lines_to_add = []
    updated_labels = []
    for line in new:
        if line.lower() in existing_lower:
            continue
        if ":" in line:
            label = line.split(":", 1)[0].strip().lower()
            matching = [
                i for i, l in enumerate(lines)
                if l.strip().lower().startswith(f"{label}:")
            ]
            if matching:
                idx = matching[0]
                lines[idx] = line
                updated_labels.append(label)
                continue
        lines_to_add.append(line)

    if updated_labels and not lines_to_add:
        _save(p, lines)
        return f"Updated existing label(s) {', '.join(updated_labels)} in {p.stem}"
    if not lines_to_add and not updated_labels:
        return f"Information already exists in {p.stem}. No duplicate added."

    first = len(lines) + 1
    lines.extend(lines_to_add)
    _save(p, lines)
    return f"Added lines {first}-{len(lines)} to {p.stem}"


def update_lines_memory_action(
    p: Path, lines: List[str], content: str, start_line: int, end_line: int
) -> str:
    _check_range(start_line, end_line, lines)
    new = content.splitlines()
    if not new:
        raise ValueError("'content' is required. Use delete_lines to remove lines")
    lines[start_line - 1 : end_line] = new
    _save(p, lines)
    return f"Updated lines {start_line}-{end_line}. Line numbers below may have shifted"


def delete_lines_memory_action(
    p: Path, lines: List[str], start_line: int, end_line: int
) -> str:
    _check_range(start_line, end_line, lines)
    del lines[start_line - 1 : end_line]
    _save(p, lines)
    return f"Deleted lines {start_line}-{end_line}. Line numbers below have shifted"


def delete_file_memory_action(p: Path) -> str:
    p.unlink()
    return f"Deleted {p.stem}"


def memory_tool(
    action: str,
    file: str = "",
    content: str = "",
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    try:
        with _LOCK:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)

            if action == "list":
                STATE.memory_listed = True
                return memory_index()

            if action in WRITES:
                STATE.memory_listed = True
            if action in WRITES and SECRET.search(content):
                return "Error: do not store secrets."

            if not file:
                return "Error: 'file' is required. Call list to see existing files"
            p = _path(file)

            if action == "create":
                return create_memory_action(p, content)

            lines = _load(p)

            if action == "read":
                return read_memory_action(p, lines, start_line, end_line)
            elif action == "append":
                return append_memory_action(p, lines, content)
            elif action == "update_lines":
                return update_lines_memory_action(p, lines, content, start_line, end_line)
            elif action == "delete_lines":
                return delete_lines_memory_action(p, lines, start_line, end_line)
            elif action == "delete_file":
                return delete_file_memory_action(p)

            return "Error: unknown action. Use list, read, create, append, update_lines, delete_lines or delete_file"

    except Exception as e:
        return f"Error: {e}"


memory_tool_spec = Tool(
    type="function",
    function=Tool.Function(
        name="memory_tool",
        description="""
Manage persistent long-term memory across conversations.
Use this tool whenever you need to memorize important facts, user preferences, ongoing project context, or data the user explicitly asked to store.
Memory can change over time: you can inspect existing memories, update lines, append new details, or overwrite outdated information.
Do not store passwords or secrets.

Actions:
- 'list': List all stored memory files.
- 'read': Read lines from a memory file (supports start_line and end_line).
- 'create': Create a new memory file with initial content (or overwrite).
- 'append': Add a new line or block to an existing memory file.
- 'update_lines': Replace specific lines in a memory file (start_line to end_line).
- 'delete_lines': Delete a line range from a memory file.
- 'delete_file': Remove a memory file completely.
""".strip(),
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "read",
                        "create",
                        "append",
                        "update_lines",
                        "delete_lines",
                        "delete_file",
                    ],
                    "description": "Memory operation to perform.",
                },
                "file": {
                    "type": "string",
                    "description": "Target memory filename (e.g. 'user_profile', 'projects', 'html_out'). Required for all actions except 'list'.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to save, append, or replace with.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line number (1-indexed) for read, update_lines, or delete_lines.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line number (inclusive) for read, update_lines, or delete_lines.",
                },
            },
            required=["action"],
        ),
    ),
)
