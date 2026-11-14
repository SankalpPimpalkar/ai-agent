from typing import Any, Dict, Optional
from src.agent.tools.base import BaseTool
from src.agent.tools.cli.operations import CliFileOperations
from src.agent.tools.cli.path_guard import CliPathGuard
from src.agent.tools.cli.validator import CliCommandValidator
from src.agent.tools.state import STATE


class CliTool(BaseTool):
    """
    Project execution and file workspace manipulation tool.
    Actions: create_file, update_file, read_file, update_lines, delete_file, list_files, run.
    """

    def __init__(
        self,
        path_guard: Optional[CliPathGuard] = None,
        command_validator: Optional[CliCommandValidator] = None,
        file_ops: Optional[CliFileOperations] = None,
    ):
        self.path_guard = path_guard or CliPathGuard()
        self.validator = command_validator or CliCommandValidator(self.path_guard)
        self.file_ops = file_ops or CliFileOperations(self.path_guard)

    @property
    def name(self) -> str:
        return "cli_tool"

    @property
    def description(self) -> str:
        return """
Execute shell commands and perform file operations safely within a project folder.
Actions: create_file, update_file, read_file, update_lines, delete_file, list_files, run.
Use 'run' to run bash commands like tests, builds, and scripts (dev servers are blocked).
Always supply 'project' and 'action'.
""".strip()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_file",
                        "update_file",
                        "read_file",
                        "update_lines",
                        "delete_file",
                        "list_files",
                        "run",
                    ],
                    "description": "Operation to perform",
                },
                "project": {
                    "type": "string",
                    "description": "Project directory name (e.g. 'todo-cli'). Do not use 'artifacts' as the project name",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file or directory path inside the project (e.g. 'todo.py' or 'data.json'). Do NOT prefix with 'artifacts/' or the project name",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content or replacement lines",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to execute for action='run'",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based starting line number for read_file or update_lines",
                },
                "end_line": {
                    "type": "integer",
                    "description": "1-based inclusive ending line number",
                },
            },
            "required": ["action"],
        }

    def run(
        self,
        action: str,
        project: str = "",
        path: str = ".",
        content: str = "",
        command: str = "",
        start_line: int = 1,
        end_line: int = 0,
        **kwargs: Any,
    ) -> str:
        try:
            action = (action or "").strip()
            proj, clean_path = self.path_guard.normalize(project, path, kwargs)

            if not proj:
                return "Error: 'project' is required"

            if action == "run":
                cmd = command or kwargs.get("cmd", "")
                if not cmd:
                    return "Error: 'command' is required for action 'run'"
                return self.validator.execute(proj, clean_path, cmd)

            if action == "create_file":
                if not content and "text" in kwargs:
                    content = kwargs["text"]
                return self.file_ops.create_file(proj, clean_path, content)

            if action == "update_file":
                if not content and "text" in kwargs:
                    content = kwargs["text"]
                return self.file_ops.update_file(proj, clean_path, content)

            if action == "read_file":
                s = int(kwargs.get("start", start_line))
                e = int(kwargs.get("end", end_line))
                return self.file_ops.read_file(proj, clean_path, s, e)

            if action == "update_lines":
                s = int(kwargs.get("start", start_line))
                e = int(kwargs.get("end", end_line))
                return self.file_ops.update_lines(proj, clean_path, content, s, e)

            if action == "delete_file":
                return self.file_ops.delete_file(proj, clean_path)

            if action in ("list_files", "list_dir", "ls"):
                return self.file_ops.list_files(proj, clean_path)

            return f"Error: unknown action '{action}'"
        except ValueError as ve:
            return f"Error: {ve}"
        except Exception as e:
            return f"Error: {e}"


cli_tool_instance = CliTool()


def cli_tool(
    action: str,
    project: str = "",
    path: str = ".",
    content: str = "",
    command: str = "",
    start_line: int = 1,
    end_line: int = 0,
    **kwargs: Any,
) -> str:
    return cli_tool_instance.run(
        action=action,
        project=project,
        path=path,
        content=content,
        command=command,
        start_line=start_line,
        end_line=end_line,
        **kwargs,
    )


cli_tool_spec = cli_tool_instance.to_spec()
