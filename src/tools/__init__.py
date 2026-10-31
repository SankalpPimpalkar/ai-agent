from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool
from .tool_builder import tool_builder
from .cli import cli_tool
from src.settings import ARTIFACTS_FOLDER_NAME

tools: List[Tool] = []
functions: Dict[str, function] = {}

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
            description=f"""
                Manage multiple projects inside {ARTIFACTS_FOLDER_NAME}. Each project is its own folder.
                Always pass 'project'. Call list_projects first if unsure which project the user means.
                Use it to scaffold projects, install packages, and create, read, update, delete or list files.
                Paths are relative to the project folder. Dangerous commands are blocked.
                Avoid 'npm run dev', use 'npm run build' to verify. Prefer update_lines for small edits instead of rewriting a whole file. read_file returns numbered lines, so read the relevant range first, then update those exact lines. Re-read before editing again, since line numbers shift after an edit.
            """,
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
                            "list_projects",
                            "list",
                        ],
                        "description": "Operation to perform",
                    },
                    "project": {
                        "type": "string",
                        "description": f"Project folder name inside {ARTIFACTS_FOLDER_NAME}, e.g. 'my-portfolio'. Required for every action except list_projects. Created automatically on first use.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or folder path relative to {ARTIFACTS_FOLDER_NAME}. For 'run', it is the working directory. Defaults to '.'",
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
                    "start_line": {
                        "type": "integer",
                        "description": "First line number (1-indexed). Used by update_lines, and optionally by read_file to read a range",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line number, inclusive. Used by update_lines and optionally by read_file. To insert before a line, set end_line to start_line - 1",
                    },
                },
                required=["action"],
            ),
        ),
    )
)

functions["tool_builder"] = tool_builder
tools.append(
    Tool(
        type="function",
        function=Tool.Function(
            name="tool_builder",
            description="""
                Create, test and run custom Python tools. Use it when no existing tool can do the job.
                Call list first to reuse an existing tool. For a new tool, call create with a function
                named exactly like the tool, plus test_code using assert statements, then call test. Only call run after the test passes. Pass run arguments as a JSON string in args. After a failed test, use read and update_lines to fix only the broken lines instead of calling create again. Once a test passes, call register with params so the tool becomes a normal tool. Always pass the same 'name' in every tool_builder call except list. If create or test fails, fix the same tool with read and update_lines. Never create a tool with a different name to work around an error.
            """,
            parameters=Tool.Function.Parameters(
                type="object",
                properties={
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "test", "run"],
                        "description": "Operation to perform",
                    },
                    "name": {
                        "type": "string",
                        "description": "Tool name in snake_case",
                    },
                    "description": {
                        "type": "string",
                        "description": "One line on what the tool does. Used by create",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code for tool.py. Used by create",
                    },
                    "test_code": {
                        "type": "string",
                        "description": "Python code for test_tool.py. Used by create",
                    },
                    "args": {
                        "type": "string",
                        "description": 'JSON object of arguments, e.g. \'{"text": "hi"}\'. Used by run',
                    },
                    "file": {
                        "type": "string",
                        "enum": ["tool", "test"],
                        "description": "Which file to read or edit. Used by read and update_lines",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line (1-indexed). Used by read and update_lines",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line, inclusive. To insert before a line, set end_line to start_line - 1",
                    },
                    "params": {
                        "type": "string",
                        "description": 'JSON object describing the tool arguments, e.g. \'{"text": {"type": "string", "description": "Text to count"}}\'. Used by register',
                    },
                },
                required=["action"],
            ),
        ),
    )
)