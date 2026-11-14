import re
from pathlib import Path
from typing import Optional, Union
from src.agent.settings import SETTINGS
from src.agent.tools.state import STATE

ROOT = SETTINGS.paths.artifacts_dir
PROJECT_NAME = re.compile(r"[A-Za-z0-9_-]+")


def safe_resolve_path(base_dir: Union[Path, str], relative_path: str) -> Path:
    """Resolve relative_path against base_dir and ensure it does not escape base_dir."""
    base = Path(base_dir).resolve()
    clean = relative_path.strip() if relative_path else "."
    full = (base / clean).resolve()
    if full != base and base not in full.parents:
        raise ValueError(f"Path escapes base directory: {relative_path}")
    return full


def _safe_path(p: str, project: str) -> Path:
    guard = CliPathGuard()
    return guard.safe_path(p, project)


class CliPathGuard:
    """Manages project directories and path boundary safety."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or ROOT

    def project_root(self, project: str) -> Path:
        proj = (project or "").strip().strip("./")
        for prefix in ("data/artifacts/", "artifacts/", "data/"):
            if proj.startswith(prefix):
                proj = proj[len(prefix):]
        if proj in ("artifacts", "data", "app", "project", ""):
            proj = getattr(STATE, "current_project", "") or "project"
        if "/" in proj:
            proj = proj.split("/", 1)[0]

        if not PROJECT_NAME.fullmatch(proj):
            raise ValueError(f"Invalid project name: {project}")
        p = (self.root / proj).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def safe_path(self, p: str, project: str) -> Path:
        base = self.project_root(project)
        clean = (p or ".").strip().removeprefix("./")
        while clean.startswith("artifacts/") or clean.startswith("data/"):
            clean = clean.split("/", 1)[1] if "/" in clean else "."
        if clean in ("artifacts", "data", ""):
            clean = "."

        proj = base.name
        if clean.startswith(f"{proj}/"):
            clean = clean[len(proj) + 1 :]
        elif clean == proj:
            clean = "."

        while clean.startswith("artifacts/") or clean.startswith("data/"):
            clean = clean.split("/", 1)[1] if "/" in clean else "."

        if clean in ("artifacts", "data", ""):
            clean = "."

        try:
            return safe_resolve_path(base, clean)
        except ValueError:
            raise ValueError(f"Path escapes project: {p}")

    def normalize(
        self, project: str, path: str, kwargs: dict
    ) -> tuple[str, str]:
        raw_proj = (
            project
            or kwargs.get("project_name", "")
            or kwargs.get("folder", "")
            or kwargs.get("dir", "")
            or getattr(STATE, "current_project", "")
        )
        proj = (raw_proj or "").strip().strip("./")
        for prefix in ("data/artifacts/", "artifacts/", "data/"):
            if proj.startswith(prefix):
                proj = proj[len(prefix):]
        if "/" in proj:
            proj = proj.split("/", 1)[0]

        clean_path = path if path not in (".", "") else kwargs.get(
            "file",
            kwargs.get("filename", kwargs.get("filepath", kwargs.get("name", "."))),
        )
        clean_path = (clean_path or ".").strip().removeprefix("./")

        # Strip any leading artifacts/ or data/ from path
        while clean_path.startswith("artifacts/") or clean_path.startswith("data/"):
            parts = clean_path.split("/", 1)
            if len(parts) == 2:
                # If path was artifacts/<project>/<file> and proj was not set, deduce proj
                sub_proj = parts[1].split("/", 1)[0] if "/" in parts[1] else ""
                if sub_proj and (not proj or proj in ("artifacts", "data", "app", "project")):
                    proj = sub_proj
                clean_path = parts[1]
            else:
                clean_path = "."

        # If clean_path contains slashes (e.g. "todo-cli/todo.py") and proj is generic, infer proj
        if "/" in clean_path:
            parts = clean_path.split("/", 1)
            first_seg = parts[0]
            if not proj or proj in ("artifacts", "data", "app", "project"):
                proj = first_seg
                clean_path = parts[1]
            elif (self.root / first_seg).is_dir():
                if proj in ("artifacts", "data", "app", "project"):
                    proj = first_seg
                clean_path = parts[1]

        if clean_path in ("artifacts", "data", ""):
            clean_path = "."

        if (not proj or proj in ("artifacts", "data", "app", "project")) and getattr(STATE, "current_project", ""):
            proj = STATE.current_project

        if (not proj or proj in ("artifacts", "app")) and self.root.exists():
            existing = [
                d.name
                for d in self.root.iterdir()
                if d.is_dir() and d.name not in ("app", "__pycache__", "artifacts")
            ]
            if len(existing) == 1:
                proj = existing[0]
            elif existing:
                proj = sorted(
                    existing,
                    key=lambda name: (self.root / name).stat().st_mtime,
                    reverse=True,
                )[0]

        if proj and clean_path.startswith(f"{proj}/"):
            clean_path = clean_path[len(proj) + 1 :]
        elif proj and clean_path == proj:
            clean_path = "."

        # Secondary check in case clean_path now has artifacts/
        while clean_path.startswith("artifacts/") or clean_path.startswith("data/"):
            clean_path = clean_path.split("/", 1)[1] if "/" in clean_path else "."

        if clean_path in ("artifacts", "data", ""):
            clean_path = "."

        return proj, clean_path
