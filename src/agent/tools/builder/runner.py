import os
import subprocess
import sys
from pathlib import Path


class ToolTestRunner:
    """Runs test suites against generated tools."""

    @staticmethod
    def run_tests(tool_dir: Path) -> tuple[bool, str]:
        test_file = tool_dir / "test_tool.py"
        if not test_file.exists():
            return False, "test_tool.py not found"

        env = {
            **os.environ,
            "PYTHONPATH": f"{tool_dir}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
        }
        res = subprocess.run(
            [sys.executable, "-m", "unittest", "test_tool.py"],
            cwd=tool_dir,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        passed = res.returncode == 0
        output = res.stderr if res.stderr else res.stdout
        return passed, output
