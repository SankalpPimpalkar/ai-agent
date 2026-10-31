# ai-agent

ai-agent is an asynchronous Python service built with FastAPI and Ollama for running local AI agents with autonomous tool execution, code generation, and sandboxed file management. The service provides endpoints for regular chat, streaming responses, and iterative tool loops equipped with loop protection safeguards, token usage accounting, dynamic tool creation via tool_builder, project-scoped workspace commands through cli_tool, and an extended mathematical evaluation engine.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Application Server (`src/main.py`)](#1-application-server-srcmainpy)
  - [2. Configuration and Settings (`src/settings.py`)](#2-configuration-and-settings-srcsettingspy)
  - [3. Data Schemas (`src/schemas.py`)](#3-data-schemas-srcschemaspy)
  - [4. Conversational Memory (`src/memory.py`)](#4-conversational-memory-srcmemorypy)
  - [5. Utilities and History Management (`src/utils.py`)](#5-utilities-and-history-management-srcutilspy)
  - [6. Tool Registry (`src/tools/__init__.py`)](#6-tool-registry-srctools__init__py)
  - [7. Dynamic Tool Builder (`src/tools/tool_builder.py`)](#7-dynamic-tool-builder-srctoolstool_builderpy)
  - [8. CLI and Project File Manager (`src/tools/cli.py`)](#8-cli-and-project-file-manager-srctoolsclipy)
  - [9. Extended Calculator Tool (`src/tools/calculator.py`)](#9-extended-calculator-tool-srctoolscalculatorpy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Clone and Environment Setup](#clone-and-environment-setup)
  - [Dependency Installation](#dependency-installation)
- [Usage Examples](#usage-examples)
  - [1. Server Startup](#1-server-startup)
  - [2. Standard Chat with Token Metrics](#2-standard-chat-with-token-metrics)
  - [3. Autonomous Tool-Assisted Execution](#3-autonomous-tool-assisted-execution)
  - [4. Event Stream Chat](#4-event-stream-chat)
- [How It Works](#how-it-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)
  - [Milestone 1: Initial Architecture and Tool Loop](#milestone-1-initial-architecture-and-tool-loop)
  - [Milestone 2: System Prompt Externalization and Documentation](#milestone-2-system-prompt-externalization-and-documentation)
  - [Milestone 3: Settings Centralization and Schema Refactoring](#milestone-3-settings-centralization-and-schema-refactoring)
  - [Milestone 4: CLI File Management and Tool Integration](#milestone-4-cli-file-management-and-tool-integration)
  - [Milestone 5: Project Isolation and Dynamic Tool Builder](#milestone-5-project-isolation-and-dynamic-tool-builder)
  - [Milestone 6: Calculator Enhancements, Loop Safeguards, and Modular Segregation](#milestone-6-calculator-enhancements-loop-safeguards-and-modular-segregation)

## Project Overview

Building reliable autonomous agents requires combining local language models with deterministic tools, bounded execution limits, and strict workspace isolation. This project demonstrates how an agent can iteratively reason, execute code, build its own missing tools, and manage file systems without entering infinite execution loops or compromising system security.

1. **Modular Code Architecture**: Separates configuration, schemas, memory, utilities, API routing, and tools into discrete modules.
2. **Bounded Execution Loop**: Limits tool calling iterations to 25 steps with automatic circuit breakers for repeated queries and consecutive errors.
3. **Dynamic Self-Extending Tooling**: Enables the agent to inspect existing capabilities, generate new Python tools with AST verification, execute test suites, and load passed tools dynamically.
4. **Sandboxed Project Operations**: Confines file creation, line edits, directory inspection, and command execution within project-scoped directories under an artifacts root.
5. **Accurate Computation**: Offloads mathematical and analytical tasks to an extended calculation module and dedicated tools.

## Key Features

- **Multi-Turn Chat and Streaming APIs**:
  - Direct JSON chat via `/api/chat` returning assistant replies alongside token usage statistics.
  - Server-Sent Events stream through `/api/stream` with `text/event-stream` media type.
  - Autonomous multi-step tool reasoning through `/api/tool`.
- **Execution Safeguards and Loop Controls**:
  - Loop boundary constrained to 25 iterations.
  - Repeated call detector halting cycles when the same function, arguments, and result occur 3 times.
  - Consecutive error breaker alerting the model when 4 consecutive tool executions fail.
  - Asynchronous thread offloading (`asyncio.to_thread`) for non-blocking tool calls.
- **Dynamic Tool Creation System (`tool_builder`)**:
  - AST-based parameter extraction and automatic JSON schema generation (`spec.json`).
  - Isolated subprocess test execution requiring a passed assertion suite (`.passed` marker).
  - Fine-grained line-level edits (`update_lines`) on tool and test code.
  - Runtime dynamic loader (`load_custom_tools`) combining pre-configured tools with tested custom tools.
- **Project-Scoped CLI Workspace (`cli_tool`)**:
  - Scoped project directories inside `artifacts/`.
  - Path traversal checks preventing escapes outside project boundaries.
  - Shell meta-character filtering and binary allowlist (`npm`, `npx`, `pnpm`, `yarn`, `git`, `node`, `pip`, `python`).
  - Line-level file modifications (`update_lines`) preserving unmodified code.
- **Extended Calculation Engine (`calculator_tool`)**:
  - 17 binary operations including `gcd`, `lcm`, `root`, `log`, percentages, and `hypot`.
  - 11 unary operations including `sqrt`, logarithms (`ln`, `log10`, `log2`), factorials, and trigonometric functions in degrees.
  - Safe error handling for divide-by-zero, domain errors, and value bounds.

## Core Components

### 1. Application Server (`src/main.py`)

Initializes the FastAPI application and HTTP endpoints. Coordinates context trimming, dynamic tool aggregation, model querying with context size `8192`, thread-dispatched function calls, loop breakers, and aggregated token accounting.

| Endpoint / Object | Implementation | Behavior |
| :--- | :--- | :--- |
| `app` | `FastAPI()` | Core application instance. |
| `/api/chat` | `@app.post("/api/chat")` | Executes single-turn chat completion and returns response content with input, output, and total token usage. |
| `/api/stream` | `@app.post("/api/stream")` | Yields real-time token chunks over HTTP `text/event-stream` using `StreamingResponse`. |
| `/api/tool` | `@app.post("/api/tool")` | Manages tool execution loop up to 25 turns, discovers custom tools, runs tool functions in worker threads, and enforces loop breakers. |

```python
@app.post("/api/tool")
async def tool_chat(query: Query):
    trim_chat()
    chat_history.append(Message(role="user", content=query.input))

    total_in = total_out = 0
    seen, content, errors = {}, "Stopped after too many steps.", 0

    for i in range(25):
        custom_tools, custom_funcs = load_custom_tools()
        all_tools = tools + custom_tools
        all_funcs = {**functions, **custom_funcs}
        formatted_msgs = [m if isinstance(m, dict) else m.model_dump() for m in chat_history]

        res: ChatResponse = await client.chat(
            model=query.model,
            messages=formatted_msgs,
            tools=all_tools,
            think=THINKING_MODEL,
            options={"num_ctx": 8192},
        )
        chat_history.append(res.message)
        total_in += res.prompt_eval_count or 0
        total_out += res.eval_count or 0

        if not res.message.tool_calls:
            content = (res.message.content or "").removeprefix("end_").strip()
            break
```

### 2. Configuration and Settings (`src/settings.py`)

Centralizes runtime environment constants, model defaults, prompt sources, and storage directory names.

| Setting | Implementation | Behavior |
| :--- | :--- | :--- |
| `DEFAULT_MODEL` | `str = "gemma4:12b"` | Default Ollama model identifier. |
| `THINKING_MODEL` | `bool = True` | Enables reasoning/thinking traces during model calls. |
| `SYSTEM_PROMPT` | `open("./system-prompt.txt").read()` | Loads base instructions, project rules, testing constraints, and tool usage guidelines. |
| `ARTIFACTS_FOLDER_NAME` | `str = "artifacts"` | Root directory where `cli_tool` manages project folders. |
| `AGENT_TOOLS_FOLDER_NAME` | `str = "agent-tools"` | Root directory where `tool_builder` stores generated tools and specs. |

```python
# MODEL CONFIGURATIONS
DEFAULT_MODEL = "gemma4:12b"
THINKING_MODEL = True
SYSTEM_PROMPT = open("./system-prompt.txt", "r").read()

# TOOLS CONFIGURATIONS
ARTIFACTS_FOLDER_NAME = "artifacts"
AGENT_TOOLS_FOLDER_NAME = "agent-tools"
```

### 3. Data Schemas (`src/schemas.py`)

Defines Pydantic models for request validation and structured message representation.

| Schema | Implementation | Behavior |
| :--- | :--- | :--- |
| `Query` | `BaseModel` | Validates API inputs, defaulting `model` to `DEFAULT_MODEL` when omitted. |
| `Message` | `BaseModel` | Stores chat messages with `role` and `content` fields. |

```python
from pydantic import BaseModel, Field
from .settings import DEFAULT_MODEL

class Query(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    input: str

class Message(BaseModel):
    role: str
    content: str
```

### 4. Conversational Memory (`src/memory.py`)

Maintains the persistent in-memory conversation list across agent turns, seeded with the externalized system prompt.

| Variable | Implementation | Behavior |
| :--- | :--- | :--- |
| `messages` | `List[Message]` | Stores active conversation turns initialized with `role="system"` containing `SYSTEM_PROMPT`. |

```python
from typing import List
from .settings import SYSTEM_PROMPT
from ollama import Message

messages: List[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
```

### 5. Utilities and History Management (`src/utils.py`)

Provides response streaming generators and sliding-window context trimming.

| Function | Implementation | Behavior |
| :--- | :--- | :--- |
| `stream_response` | `async def stream_response(...)` | Consumes chunks from `client.chat(..., stream=True)`, yields token strings, and appends the complete assistant response to memory. |
| `_role` | `def _role(m)` | Extracts role attribute whether the item is a dictionary or an object instance. |
| `trim_chat` | `def trim_chat(max_messages=20)` | Retains `messages[0]` (system prompt) and the latest $N$ turns, popping leading turns until the first post-system turn has `role="user"`. |

```python
def trim_chat(max_messages: int = 20):
    system_prompt = messages[0]
    chat = messages[1:]
    if len(chat) > max_messages:
        chat = chat[-max_messages:]
        while chat and _role(chat[0]) != "user":
            chat.pop(0)
    messages[:] = [system_prompt] + chat
```

### 6. Tool Registry (`src/tools/__init__.py`)

Assembles the static tool specifications and runtime dispatch dictionary passed into Ollama requests.

| Registry Element | Implementation | Behavior |
| :--- | :--- | :--- |
| `tools` | `List[Tool]` | Contains Ollama `Tool` declarations for `calculator_tool`, `cli_tool`, and `tool_builder`. |
| `functions` | `Dict[str, function]` | Maps tool names to executable Python functions for runtime dispatch. |

```python
from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool
from .tool_builder import tool_builder
from .cli import cli_tool
from src.settings import ARTIFACTS_FOLDER_NAME

tools: List[Tool] = []
functions: Dict[str, function] = {}

functions["calculator_tool"] = calculator_tool
...
functions["cli_tool"] = cli_tool
...
functions["tool_builder"] = tool_builder
```

### 7. Dynamic Tool Builder (`src/tools/tool_builder.py`)

Enables the agent to generate, inspect, test, and register custom Python tools on disk under `agent-tools/`.

| Action | Implementation | Behavior |
| :--- | :--- | :--- |
| `list` | `tool_builder(action="list")` | Lists custom tools with test status (`tested` or `untested`) and descriptions. |
| `create` | `tool_builder(action="create", ...)` | Parses Python code via `ast.parse`, extracts parameters and docstrings, saves `tool.py`, `test_tool.py`, `spec.json`, and `description.txt`. |
| `test` | `tool_builder(action="test", ...)` | Executes `test_tool.py` in an isolated subprocess; touches `.passed` on zero returncode or unlinks it on failure. |
| `run` | `tool_builder(action="run", ...)` | Runs the custom tool function with JSON arguments in a subprocess. |
| `read` | `tool_builder(action="read", ...)` | Reads numbered line ranges from `tool.py` or `test_tool.py`. |
| `update_lines` | `tool_builder(action="update_lines", ...)` | Replaces specified line range, reparses AST to regenerate `spec.json`, and clears `.passed`. |
| `load_custom_tools` | `def load_custom_tools()` | Discovers all tools in `agent-tools/` with `.passed` markers, building Ollama `Tool` schemas and runner functions. |

```python
def load_custom_tools():
    tools, funcs = [], {}
    for d in sorted(TOOLS_DIR.iterdir()) if TOOLS_DIR.exists() else []:
        spec_file = d / "spec.json"
        if d.is_dir() and spec_file.exists() and (d / ".passed").exists():
            spec = json.loads(spec_file.read_text(encoding="utf-8"))
            tools.append(
                Tool(
                    type="function",
                    function=Tool.Function(
                        name=d.name,
                        description=(d / "description.txt").read_text(encoding="utf-8").strip(),
                        parameters=Tool.Function.Parameters(
                            type="object",
                            properties=spec["properties"],
                            required=spec.get("required", list(spec["properties"])),
                        ),
                    ),
                )
            )
            funcs[d.name] = custom_tool_runner(d.name)
    return tools, funcs
```

### 8. CLI and Project File Manager (`src/tools/cli.py`)

Provides project-isolated workspace management inside `artifacts/<project>`. Enforces strict path safety and command allowlists.

| Action | Implementation | Behavior |
| :--- | :--- | :--- |
| `list_projects` | `cli_tool(action="list_projects")` | Lists directory names of existing projects inside `artifacts/`. |
| `create_file` | `cli_tool(action="create_file", ...)` | Writes file content, creating missing parent directories. |
| `read_file` | `cli_tool(action="read_file", ...)` | Returns numbered file lines with optional `start_line` and `end_line` slicing. |
| `update_file` | `cli_tool(action="update_file", ...)` | Overwrites or appends content to a project file. |
| `update_lines` | `cli_tool(action="update_lines", ...)` | Replaces 1-indexed line ranges in place without modifying surrounding lines. |
| `delete_file` | `cli_tool(action="delete_file", ...)` | Deletes a file inside the project directory. |
| `create_folder` | `cli_tool(action="create_folder", ...)` | Creates directories inside project directory. |
| `delete_folder` | `cli_tool(action="delete_folder", ...)` | Deletes a subfolder; blocks deletion of the project root. |
| `list` | `cli_tool(action="list", ...)` | Lists files and subdirectories at specified project path. |
| `run` | `cli_tool(action="run", ...)` | Executes commands matching allowed binaries (`npm`, `npx`, `pnpm`, `yarn`, `git`, `node`, `pip`, `python`) with `CI=1` and `shell=False`. |

```python
def _safe_path(p: str, project: str) -> Path:
    base = _project_root(project)
    full = (base / p).resolve()
    if full != base and base not in full.parents:
        raise ValueError(f"Path escapes project: {p}")
    return full

def _validate_command(command: str) -> list[str]:
    if SHELL_META.search(command):
        raise ValueError("Shell operators are not allowed")

    parts = command.strip().split()
    if not parts:
        raise ValueError("Empty command")

    name = Path(parts[0]).name.lower()
    if name in BLACK_LISTED:
        raise ValueError(f"Blacklisted command: {name}")
    if name not in ALLOWED_BINARIES:
        raise ValueError(f"Command not allowed: {name}")

    for arg in parts[1:]:
        if arg.lower() in BLACK_LISTED:
            raise ValueError(f"Blacklisted argument: {arg}")
        if Path(arg).is_absolute() or ".." in re.split(r"[\\/]", arg):
            raise ValueError(f"Unsafe path argument: {arg}")
    return parts
```

### 9. Extended Calculator Tool (`src/tools/calculator.py`)

Executes mathematical evaluation across unary and binary operators with numeric formatting and error protection.

| Operation Category | Operators | Behavior |
| :--- | :--- | :--- |
| Binary Operations (`BINARY`) | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `max`, `min`, `gcd`, `lcm`, `root`, `log`, `percent_of`, `percent_change`, `round`, `hypot` | Evaluates two operands `op1` and `op2`. Computes arbitrary base logarithms, $b$-th roots, percentages, and greatest common divisors. |
| Unary Operations (`UNARY`) | `sqrt`, `abs`, `ln`, `log10`, `log2`, `ceil`, `floor`, `factorial`, `sin`, `cos`, `tan` | Evaluates single operand `op1`. Calculates trigonometric values from angles in degrees and computes factorials up to 1000. |

```python
def calculator_tool(op1: float, operator: str, op2: float = 0.0) -> str:
    """
    Use when the user asks for a calculation. Always use this instead of doing math yourself.
    """
    try:
        a, b = float(op1), float(op2)
        if operator in BINARY:
            return _fmt(BINARY[operator](a, b))
        if operator in UNARY:
            return _fmt(UNARY[operator](a))
        return f"Error: unsupported operator. Use one of: {', '.join(list(BINARY) + list(UNARY))}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, OverflowError):
        return "Error: input is outside the valid range for this operation"
    except Exception:
        return "Error: invalid input"
```

## Project Structure

```text
ai-agent/
├── Makefile                          # Development task automation commands
├── README.md                         # Project documentation
├── requirements.txt                  # Python package dependencies
├── system-prompt.txt                 # Agent system prompt and operational constraints
├── agent-tools/                      # Storage directory for custom-built tools
│   ├── days_until/                   # Custom tool: calculates days until target date
│   │   ├── description.txt           # Tool description used in LLM tool definition
│   │   ├── spec.json                 # JSON schema of tool parameters
│   │   ├── test_tool.py              # Test suite for days_until
│   │   └── tool.py                   # Python implementation of days_until
│   ├── password_generator/           # Custom tool: generates passwords with entropy metrics
│   │   ├── description.txt           # Tool description used in LLM tool definition
│   │   ├── spec.json                 # JSON schema of tool parameters
│   │   ├── test_tool.py              # Test suite for password_generator
│   │   └── tool.py                   # Python implementation of password_generator
│   └── sha256/                       # Custom tool: computes SHA-256 hashes
│       ├── description.txt           # Tool description used in LLM tool definition
│       ├── spec.json                 # JSON schema of tool parameters
│       ├── test_tool.py              # Test suite for sha256
│       └── tool.py                   # Python implementation of sha256
└── src/
    ├── __init__.py                   # Root package marker
    ├── main.py                       # FastAPI application, tool loop, and HTTP route handlers
    ├── memory.py                     # Global chat history storage and initialization
    ├── schemas.py                    # Pydantic input and output request schemas
    ├── settings.py                   # Model configuration, prompts, and folder constants
    ├── utils.py                      # Streaming generators and conversation trimming logic
    └── tools/
        ├── __init__.py               # Tool registry importing static tools and schemas
        ├── calculator.py             # Extended arithmetic, unary, and binary math tool
        ├── cli.py                    # Project-scoped CLI runner and safe file manager
        └── tool_builder.py           # Custom tool generator, AST parser, and test runner
```

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Ollama service running locally with a tool-supporting model (configured default: `gemma4:12b` or alternative like `llama3.2`)

Install Ollama:

- macOS (via Homebrew):
  ```bash
  brew install ollama
  ```
- Linux:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- Windows:
  Download and run the installer from the official Ollama website.

Pull a model:

```bash
ollama pull gemma4:12b
```

### Clone and Environment Setup

```bash
git clone https://github.com/SankalpPimpalkar/ai-agent.git
cd ai-agent
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (Command Prompt):

```cmd
.venv\Scripts\activate
```

### Dependency Installation

Install dependencies from `requirements.txt` and ensure `ollama` is installed:

```bash
pip install -r requirements.txt
pip install ollama
```

## Usage Examples

### 1. Server Startup

Run the development server via Make or Uvicorn:

```bash
make dev
```

Or using Uvicorn directly:

```bash
uvicorn src.main:app --reload --reload-dir src
```

### 2. Standard Chat with Token Metrics

Query `/api/chat` to receive assistant replies and token consumption metrics:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Explain the concept of an AI agent in brief."
  }'
```

### 3. Autonomous Tool-Assisted Execution

Query `/api/tool` to execute multi-step requests involving math calculations, project scaffolding, or dynamic tool construction:

```bash
curl -X POST http://localhost:8000/api/tool \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Calculate the hypotenuse of a right triangle with sides 9 and 12, then calculate 15 percent of 400."
  }'
```

### 4. Event Stream Chat

Stream token chunks directly from `/api/stream`:

```bash
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "input": "Outline four steps to test an LLM tool builder."
  }'
```

## How It Works

1. **Context Initialization and Normalization**: In-memory history in `src/memory.py` starts with `SYSTEM_PROMPT`. On each request, `trim_chat` truncates history to the latest turns while ensuring the post-system message is a user message.
   ```python
   def trim_chat(max_messages: int = 20):
       system_prompt = messages[0]
       chat = messages[1:]
       if len(chat) > max_messages:
           chat = chat[-max_messages:]
           while chat and _role(chat[0]) != "user":
               chat.pop(0)
       messages[:] = [system_prompt] + chat
   ```

2. **Dynamic Tool Resolution**: At every iteration in `/api/tool`, `load_custom_tools` scans `agent-tools/` for validated custom tools and aggregates them with static tools.
   ```python
   custom_tools, custom_funcs = load_custom_tools()
   all_tools = tools + custom_tools
   all_funcs = {**functions, **custom_funcs}
   ```

3. **Model Request with Context Budget**: The server submits the conversation and tools to `client.chat` with `think=THINKING_MODEL` and `options={"num_ctx": 8192}`.
   ```python
   res: ChatResponse = await client.chat(
       model=query.model,
       messages=formatted_msgs,
       tools=all_tools,
       think=THINKING_MODEL,
       options={"num_ctx": 8192},
   )
   ```

4. **Thread-Safe Dispatch and Circuit Breakers**: If tool calls are returned, functions are executed in worker threads via `asyncio.to_thread`. Execution output is monitored:
   - If an identical call repeats 3 times, an instruction stops looping.
   - If 4 consecutive tool errors occur, the loop halts with an error notice.
   ```python
   result = await asyncio.to_thread(all_funcs[name], **args)
   ...
   errors = errors + 1 if result.startswith("Error") else 0
   if errors >= 4:
       result += "\nToo many errors in a row. Stop and tell the user what is failing."

   key = (name, json.dumps(args, sort_keys=True, default=str), result)
   seen[key] = seen.get(key, 0) + 1
   if seen[key] >= 3:
       result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."
   ```

5. **Loop Termination**: When the model finishes tool calls or the loop reaches the 25-turn cap, the final response is returned with token usage totals.

## Project Roadmap & Commit History

### Milestone 1: Initial Architecture and Tool Loop

- `5271013`: Add initial project structure with core functionality and dependencies
  - Initialized git repository with `.gitignore` and `requirements.txt`.
  - Implemented initial FastAPI endpoints (`/api/chat`, `/api/tool`, `/api/stream`) in `src/main.py`.
  - Set up tool registration and basic arithmetic tool in `src/tools/`.

### Milestone 2: System Prompt Externalization and Documentation

- `373f2a4`: docs: add comprehensive project README and externalize system prompt
  - Extracted embedded system prompt to standalone `system-prompt.txt`.
  - Added comprehensive repository documentation adhering to markdown standards.

### Milestone 3: Settings Centralization and Schema Refactoring

- `38b997e`: refactor: move DEFAULT_MODEL and SYSTEM_PROMPT to settings.py; update Query model to use DEFAULT_MODEL
  - Created `src/settings.py` for global configurations (`DEFAULT_MODEL`, `SYSTEM_PROMPT`).
  - Updated Pydantic request models in `src/schemas.py` to use settings defaults.

### Milestone 4: CLI File Management and Tool Integration

- `ec638f8`: feat: enhance chat functionality and add CLI tool for file management
  - Created `src/tools/cli.py` for file creation, reading, and command execution.
  - Registered `cli_tool` into the agent tool list.

### Milestone 5: Project Isolation and Dynamic Tool Builder

- `bae7c48`: Refactor CLI tool to support project-based file management and enhance functionality
  - Added project directory scoping inside `artifacts/` with path safety validation.
  - Implemented line-level file updates (`update_lines`) and command execution safety rules.
  - Created `src/tools/tool_builder.py` allowing the agent to write, test, read, and run custom Python tools.
  - Added `Makefile` for streamlined development execution (`make dev`).

### Milestone 6: Calculator Enhancements, Loop Safeguards, and Modular Segregation

- `442762d`: Enhance calculator tool with additional operations and error handling
  - Expanded `src/tools/calculator.py` with 17 binary operations and 11 unary operations.
  - Replaced unbounded tool loops with a 25-turn bounded loop in `src/main.py`.
  - Added loop protection mechanisms: repeated call breaker (`seen >= 3`) and consecutive error limiter (`errors >= 4`).
  - Added dynamic tool loader (`load_custom_tools`) in `src/tools/tool_builder.py` scanning `agent-tools/`.
  - Segregated conversation state into `src/memory.py` and helper functions into `src/utils.py`.
