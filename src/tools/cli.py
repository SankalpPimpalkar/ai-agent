# file_manager_tool.py
import re
import shutil
import subprocess
from pathlib import Path

ROOT = (Path.cwd() / "generated-sites").resolve()

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

# Only these binaries can run, anything else is rejected
ALLOWED_BINARIES = {"npm", "npx", "pnpm", "yarn", "git", "node", "pip", "python"}

# Blocks chaining, redirection, substitution
SHELL_META = re.compile(r"[;&|`$<>\n\r]")


def _safe_path(p: str) -> Path:
    full = (ROOT / p).resolve()
    if full != ROOT and ROOT not in full.parents:
        raise ValueError(f"Path escapes generated-sites: {p}")
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
    path: str = ".",
    content: str = "",
    command: str = "",
    append: bool = False,
) -> str:
    ROOT.mkdir(parents=True, exist_ok=True)

    try:
        if action == "run":
            cwd = _safe_path(path)
            cwd.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                _validate_command(command),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
                shell=False,
            )
            if result.returncode != 0:
                return f"Error: {result.stderr or result.stdout}"
            return result.stdout or "Command completed"

        if action == "create_file":
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Created {path}"

        if action == "read_file":
            return _safe_path(path).read_text(encoding="utf-8")

        if action == "update_file":
            p = _safe_path(path)
            with open(p, "a" if append else "w", encoding="utf-8") as f:
                f.write(content)
            return f"Updated {path}"

        if action == "delete_file":
            _safe_path(path).unlink()
            return f"Deleted {path}"

        if action == "create_folder":
            _safe_path(path).mkdir(parents=True, exist_ok=True)
            return f"Created folder {path}"

        if action == "delete_folder":
            p = _safe_path(path)
            if p == ROOT:
                raise ValueError("Cannot delete the generated-sites root")
            shutil.rmtree(p)
            return f"Deleted folder {path}"

        if action == "list":
            entries = sorted(_safe_path(path).iterdir())
            return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)

        return f"Error: Unknown action '{action}'"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {e}"
