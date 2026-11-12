from pathlib import Path
from typing import List, Union


def clip_text(text: str, max_chars: int = 2000, suffix_only: bool = True) -> str:
    """
    Truncate text if it exceeds max_chars.
    If suffix_only is True, keeps the last max_chars prefixed with '...(truncated)\\n'.
    If suffix_only is False, keeps the first max_chars suffixed with '\\n...(truncated)'.
    """
    if len(text) <= max_chars:
        return text
    if suffix_only:
        return "...(truncated)\n" + text[-max_chars:]
    return text[:max_chars] + "\n...(truncated)"


def safe_resolve_path(base_dir: Union[Path, str], relative_path: str) -> Path:
    """
    Resolve relative_path against base_dir and ensure it does not escape base_dir.
    """
    base = Path(base_dir).resolve()
    full = (base / relative_path).resolve()
    if full != base and base not in full.parents:
        raise ValueError(f"Path escapes base directory: {relative_path}")
    return full


def read_numbered_lines(
    lines_or_text: Union[List[str], str],
    start_line: int = 1,
    end_line: int = 0,
) -> str:
    """
    Format lines with 1-based line numbers 'i: line'.
    start_line is 1-indexed. end_line is inclusive (0 means till the end).
    """
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
    """
    Replace a range of lines (1-indexed, inclusive) in original_text with new_content.
    Raises ValueError on invalid line ranges.
    """
    lines = original_text.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line - 1 or start_line > len(lines) + 1:
        raise ValueError(
            f"Invalid range {start_line}-{end_line}, file has {len(lines)} lines"
        )

    new_lines = new_content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    lines[start_line - 1 : end_line] = new_lines
    return "".join(lines)

