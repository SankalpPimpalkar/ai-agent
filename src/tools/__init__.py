from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool
from .cli import cli_tool

tools: List[Tool] = []
functions: Dict[str, function] = {}

# Calculator Tool
functions["calculator_tool"] = calculator_tool
tools.append(
    Tool(
        type="function",
        function=Tool.Function(
            name="calculator_tool",
            description="Perform basic arithmetic on two numbers. After final operations explain the answer naturally",
            parameters=Tool.Function.Parameters(
                type="object",
                properties={
                    "op1": {"type": "float", "description": "First Operand"},
                    "op2": {"type": "float", "description": "Second Operand"},
                    "operator": {
                        "type": "string",
                        "enum": list(_OPS.keys()),
                        "description": "Operator",
                    },
                },
                required=["op1", "op2", "operator"],
            ),
        ),
    )
)

functions["cli_tool"] = cli_tool
tools.append(
    Tool(
        type="function",
        function=Tool.Function(
            name="cli_tool",
            description="""
                Manage files and folders and run safe CLI commands (npm, npx, pnpm, yarn, git, node, pip, python)
                inside the generated-sites directory. Use it to scaffold projects, install packages, and create,
                read, update, delete or list files and folders. All paths are relative to generated-sites.
                Dangerous commands are blocked. After final operations explain the result naturally
            """
            ,
            parameters=Tool.Function.Parameters(
                type="object",
                properties={
                    "action": {
                        "type": "string",
                        "enum": [
                            "run",
                            "create_file",
                            "read_file",
                            "update_file",
                            "delete_file",
                            "create_folder",
                            "delete_folder",
                            "list",
                        ],
                        "description": "Operation to perform",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or folder path relative to generated-sites. For 'run', it is the working directory. Defaults to '.'",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content. Used by create_file and update_file",
                    },
                    "command": {
                        "type": "string",
                        "description": "CLI command to run, e.g. 'npm install axios'. Used only by 'run'",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, update_file appends to the file instead of overwriting it",
                    },
                },
                required=["action"],
            ),
        ),
    )
)
