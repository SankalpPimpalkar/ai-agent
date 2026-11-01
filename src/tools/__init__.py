from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool
from .cli import cli_tool
from .memory import memory_tool
from src.settings import SETTINGS
from .tool_builder import tool_builder

ARTIFACTS_FOLDER_NAME = SETTINGS.ARTIFACTS_FOLDER_NAME

tools: List[Tool] = []
functions: Dict[str, function] = {}

functions["web_search"] = SETTINGS.OLLAMA_CLIENT.web_search
web_search_spec = Tool(
    type="function",
    function=Tool.Function(
        name="web_search",
        description="""
                Search the web for current information. Use this for recent events,
                facts you are unsure about, or anything that may have changed.
                Returns a list of results with title, url and content snippet.
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "query": {
                    "type": "string",
                    "description": "Search query. Keep it short and specific.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                },
            },
            required=["query"],
        ),
    ),
)
tools.append(web_search_spec)

functions["web_fetch"] = SETTINGS.OLLAMA_CLIENT.web_fetch
web_fetch_spec = Tool(
    type="function",
    function=Tool.Function(
        name="web_fetch",
        description="""
                Fetch the full content of a web page from a URL.
                Use this after web_search when the snippet is not enough,
                or when the user gives you a URL.
                Returns the page title, content and links.
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "url": {
                    "type": "string",
                    "description": "Full URL including https://",
                },
            },
            required=["url"],
        ),
    ),
)
tools.append(web_fetch_spec)

functions["calculator_tool"] = calculator_tool
calculator_tool_spec = Tool(
    type="function",
    function=Tool.Function(
        name="calculator_tool",
        description="""
                Perform math calculations. Always use this instead of calculating in your head.
                Two-number operations (op1, operator, op2): + - * / // % ** max min gcd lcm hypot,
                root (op2-th root of op1), log (op1 with base op2), percent_of (op1 percent of op2),
                percent_change (change from op1 to op2), round (op1 to op2 decimals).
                One-number operations (op1, operator, no op2): sqrt abs ln log10 log2 ceil floor factorial,
                sin cos tan (angle in degrees).
                For longer expressions, call it step by step and use each result in the next call.
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "op1": {"type": "float", "description": "First Operand"},
                "op2": {
                    "type": "float",
                    "description": "Second number. Omit for one-number operations like sqrt or factorial",
                },
                "operator": {
                    "type": "string",
                    "enum": _OPS,
                    "description": "Operation to apply. Two-number: + - * / // % ** max min gcd lcm root (op2-th root of op1) log (op1 with base op2) percent_of (op1 percent of op2) percent_change (from op1 to op2) round (op1 to op2 decimals) hypot. One-number (op2 ignored): sqrt abs ln log10 log2 ceil floor factorial sin cos tan (degrees).",
                },
            },
            required=["op1", "operator"],
        ),
    ),
)
tools.append(calculator_tool_spec)

functions["tool_builder"] = tool_builder
tool_builder_spec = Tool(
    type="function",
    function=Tool.Function(
        name="tool_builder",
        description="""
                Build a new reusable Python tool when no existing tool fits the request. Call list first. Steps: create, then test. Once the test passes, the tool is available and you call it directly. Always pass the same name. Fix failures with read and update_lines, never by creating a tool with a different name. The function must be named exactly like the tool, have type hints, and a docstring whose first line starts with "Use when the user asks to ...". 
                
                Example create call: {"action": "create", "name": "word_count", "code": "def word_count(text: str) -> str:\n    \"\"\"Use when the user asks to count words.\"\"\"\n    return str(len(text.split()))", "test_code": "assert word_count('a b c') == '3'"}
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "action": {
                    "type": "string",
                    "enum": [
                        "list",
                        "create",
                        "test",
                        "run",
                        "read",
                        "update_lines",
                    ],
                    "description": "Operation to perform",
                },
                "name": {
                    "type": "string",
                    "description": "Tool name in snake_case. Required for every action except list",
                },
                "description": {
                    "type": "string",
                    "description": "Optional. Defaults to the first line of the function docstring. Used by create",
                },
                "code": {
                    "type": "string",
                    "description": "Python code. For create it is the full tool.py. For update_lines it is the replacement lines",
                },
                "test_code": {
                    "type": "string",
                    "description": "Python test code using assert statements. Used by create. The import of the tool is added automatically",
                },
                "run_args": {
                    "type": "string",
                    "description": "Only for action 'run'. JSON object with the tool's arguments, e.g. '{\"text\": \"hi\"}'. Never use it for create",
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
            },
            required=["action"],
        ),
    ),
)
tools.append(tool_builder_spec)

functions["memory_tool"] = memory_tool
memory_tool_spec = Tool(
    type="function",
    function=Tool.Function(
        name="memory_tool",
        description="""
                Manage long-term memory as small topic files, for example user_profile
                or topics_discussed. Call list first to see existing files. Create a new
                file only when no existing one fits. Line 1 of every file is a short
                description. Read a file before update_lines or delete_lines, because
                line numbers shift after edits. Never store passwords or secrets.
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "action": {
                    "type": "string",
                    "description": "list, read, create, append, update_lines, delete_lines or delete_file",
                },
                "file": {
                    "type": "string",
                    "description": "File name without extension, lowercase",
                },
                "content": {
                    "type": "string",
                    "description": "Text to write, one fact per line",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line (inclusive)",
                },
            },
            required=["action"],
        ),
    ),
)
tools.append(memory_tool_spec)


functions["cli_tool"] = cli_tool
cli_tool_spec = Tool(
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
tools.append(cli_tool_spec)
