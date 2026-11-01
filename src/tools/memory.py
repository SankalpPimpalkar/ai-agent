import re
import threading
from src.settings import BASE_DIR, SETTINGS

MEMORY_DIR = BASE_DIR / SETTINGS.MEMORY_FOLDER_NAME
MAX_FILES = 50
MAX_LINES = 200
_LOCK = threading.Lock()
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def _path(name: str):
    name = name.strip().lower().removesuffix(".md")
    if not _NAME.match(name):
        raise ValueError("Invalid file name. Use lowercase letters, numbers, '-' or '_'")
    return MEMORY_DIR / f"{name}.md"


def _load(p) -> list[str]:
    if not p.exists():
        raise ValueError(f"'{p.stem}' does not exist. Call list or create it first")
    return p.read_text(encoding="utf-8").splitlines()


def _save(p, lines: list[str]):
    if len(lines) > MAX_LINES:
        raise ValueError(f"File is full ({MAX_LINES} lines max). Merge or delete lines first")
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
                return memory_index()

            if not file:
                return "Error: 'file' is required. Call list to see existing files"
            p = _path(file)

            def check_range(lines):
                if not 1 <= start_line <= end_line <= len(lines):
                    raise ValueError(
                        f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines"
                    )

            if action == "create":
                if p.exists():
                    raise ValueError(f"'{p.stem}' already exists. Use append or update_lines")
                if len(list(MEMORY_DIR.glob("*.md"))) >= MAX_FILES:
                    raise ValueError(f"Too many files ({MAX_FILES} max). Merge or delete some")
                lines = content.splitlines()
                if not lines:
                    raise ValueError("'content' is required. Line 1 should describe the file")
                _save(p, lines)
                return f"Created {p.stem}"

            lines = _load(p)

            if action == "read":
                if not lines:
                    return f"{p.stem} is empty"
                start = max(start_line, 1)
                end = min(end_line or len(lines), len(lines))
                if start > end:
                    raise ValueError(f"Invalid range {start}-{end}, file has {len(lines)} lines")
                return "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))

            if action == "append":
                new = content.splitlines()
                if not new:
                    raise ValueError("'content' is required")
                first = len(lines) + 1
                lines.extend(new)
                _save(p, lines)
                return f"Added lines {first}-{len(lines)} to {p.stem}"

            if action == "update_lines":
                check_range(lines)
                new = content.splitlines()
                if not new:
                    raise ValueError("'content' is required. Use delete_lines to remove lines")
                lines[start_line - 1:end_line] = new
                _save(p, lines)
                return f"Updated lines {start_line}-{end_line}. Line numbers below may have shifted"

            if action == "delete_lines":
                check_range(lines)
                del lines[start_line - 1:end_line]
                _save(p, lines)
                return f"Deleted lines {start_line}-{end_line}. Line numbers below have shifted"

            if action == "delete_file":
                p.unlink()
                return f"Deleted {p.stem}"

            return "Error: unknown action. Use list, read, create, append, update_lines, delete_lines or delete_file"

    except Exception as e:
        return f"Error: {e}"