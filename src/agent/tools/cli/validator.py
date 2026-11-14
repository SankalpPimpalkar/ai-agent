import os
import re
import subprocess
from src.agent.tools.cli.path_guard import CliPathGuard
from src.agent.tools.state import STATE

DEV_SERVERS = ["npm run dev", "npm start", "yarn dev", "pnpm dev"]


def clip_text(text: str, max_chars: int = 2000, suffix_only: bool = True) -> str:
    if len(text) <= max_chars:
        return text
    if suffix_only:
        return "...(truncated)\n" + text[-max_chars:]
    return text[:max_chars] + "\n...(truncated)"


class CliCommandValidator:
    """Validates commands, guards against dev servers, and runs subprocesses safely."""

    def __init__(self, path_guard: CliPathGuard):
        self.path_guard = path_guard

    def execute(self, project: str, path: str, command: str) -> str:
        cwd = self.path_guard.safe_path(path, project)
        cwd.mkdir(parents=True, exist_ok=True)

        cmd_strip = command.strip()
        if not cmd_strip:
            raise ValueError("Empty command")

        cmd_lower = cmd_strip.lower()
        for ds in DEV_SERVERS:
            if cmd_lower.startswith(ds):
                raise ValueError("Dev servers never exit. Use 'npm run build' to verify")

        if STATE.main_file:
            for generic in ("app.py", "main.py", "run.py", "script.py"):
                if (
                    generic != STATE.main_file
                    and generic in cmd_strip
                    and not (cwd / generic).exists()
                ):
                    cmd_strip = re.sub(
                        r"\b" + re.escape(generic) + r"\b", STATE.main_file, cmd_strip
                    )

        env = {
            **os.environ,
            "CI": "1",
            "PYTHONPATH": f"{cwd}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
        }

        try:
            result = subprocess.run(
                cmd_strip,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
                shell=True,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            if result.returncode != 0:
                return f"Error: {clip_text(result.stderr or result.stdout)}"
            return clip_text(result.stdout) or "Command completed"
        except subprocess.TimeoutExpired:
            return f"Error: Command '{cmd_strip}' timed out after 15 seconds. Ensure loops terminate and pointers advance."
        except Exception as e:
            return f"Error: {e}"
