from typing import Dict, List, Optional


class TaskState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.memory_listed = False
        self.tools_listed = False
        self.creates = 0
        self.current_project = ""
        # Mapping of filename -> relative path inside project (e.g. "todo.py": "todo.py")
        self.files: Dict[str, str] = {}
        # Primary script entrypoint (e.g. "todo.py", "converter.py", "db.py")
        self.main_file: str = ""

    def register_file(self, path: str):
        clean = path.strip().lstrip("./")
        filename = clean.split("/")[-1]
        self.files[filename] = clean
        if filename.endswith(".py") and (
            not self.main_file or filename not in ("test.py", "tests.py")
        ):
            self.main_file = filename

    def unregister_file(self, path: str):
        clean = path.strip().lstrip("./")
        filename = clean.split("/")[-1]
        self.files.pop(filename, None)
        if self.main_file == filename:
            py_files = [f for f in self.files if f.endswith(".py")]
            self.main_file = py_files[0] if py_files else ""

    def resolve_file(self, requested_path: str) -> str:
        """
        Resolves a file path, preventing hallucination between filenames (e.g. app.py vs todo.py).
        """
        if not requested_path or requested_path in (".", ""):
            return self.main_file or requested_path
        clean = requested_path.strip().lstrip("./")
        filename = clean.split("/")[-1]
        if filename in self.files:
            return self.files[filename]
        # Model hallucinated a generic name instead of the actual script created
        if filename in ("app.py", "main.py", "script.py", "run.py") and self.main_file:
            return self.main_file
        return clean

    def get_file_summary(self) -> str:
        if not self.files:
            return "No files created yet"
        return ", ".join(f"{fname}" for fname in sorted(self.files.keys()))


STATE = TaskState()
