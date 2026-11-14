import json
from pathlib import Path
from typing import Optional
from src.agent.settings import SETTINGS

TOOLS_DIR = SETTINGS.paths.tools_dir
FILES = {"tool": "tool.py", "test": "test_tool.py"}


class ToolFileStore:
    """Manages file storage and directory operations for custom tools."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or TOOLS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def tool_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_file(self, name: str, file_type: str) -> str:
        f = self.tool_dir(name) / FILES[file_type]
        if not f.exists():
            raise FileNotFoundError(f"File {FILES[file_type]} not found for tool {name}")
        return f.read_text(encoding="utf-8")

    def write_file(self, name: str, file_type: str, content: str) -> None:
        f = self.tool_dir(name) / FILES[file_type]
        f.write_text(content, encoding="utf-8")

    def save_spec(
        self, name: str, props: dict, required: list, description: str
    ) -> None:
        spec = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }
        (self.tool_dir(name) / "spec.json").write_text(
            json.dumps(spec, indent=2), encoding="utf-8"
        )
        (self.tool_dir(name) / "description.txt").write_text(
            description, encoding="utf-8"
        )

    def mark_passed(self, name: str) -> None:
        (self.tool_dir(name) / ".passed").touch()

    def is_passed(self, name: str) -> bool:
        return (self.tool_dir(name) / ".passed").exists()
