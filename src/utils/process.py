import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def run_subprocess(
    cmd: List[str],
    cwd: Optional[Union[Path, str]] = None,
    timeout: int = 60,
    env: Optional[Dict[str, str]] = None,
    max_output_chars: int = 2000,
) -> Tuple[int, str]:
    """
    Run a subprocess command securely without shell=True.
    Returns (returncode, captured_output).
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    combined = (result.stdout + result.stderr).strip()
    if max_output_chars and len(combined) > max_output_chars:
        combined = combined[-max_output_chars:]
    return result.returncode, combined

