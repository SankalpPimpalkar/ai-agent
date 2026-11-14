# ai-agent

ai-agent is an asynchronous Python service built with FastAPI and Ollama for running local AI agents with autonomous tool execution, persistent categorized memory, database prompt version tracking, live web search, code generation, and sandboxed file management. The service provides endpoints for regular chat, token streaming responses, and iterative tool loops equipped with circuit breakers, token usage accounting, dynamic tool creation via tool_builder, project-scoped workspace commands through cli_tool, background memory consolidation, and an Alembic-managed database tracking prompt configurations.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Application API Server (`src/main.py`)](#1-application-api-server-srcmainpy)
  - [2. Autonomous Agent Engine (`src/agent.py`)](#2-autonomous-agent-engine-srcagentpy)
  - [3. Centralized Configuration (`src/settings.py`)](#3-centralized-configuration-srcsettingspy)
  - [4. Long-Term Memory & Message Assembly (`src/memory.py`)](#4-long-term-memory--message-assembly-srcmemorypy)
  - [5. Database Layer & Prompt Versioning (`src/db/models.py`)](#5-database-layer--prompt-versioning-srcdbmodelspy)
  - [6. Utilities and Lifespan Initialization (`src/utils.py`)](#6-utilities-and-lifespan-initialization-srcutilspy)
  - [7. Tool Registry and Dynamic Dispatch (`src/tools/__init__.py`)](#7-tool-registry-and-dynamic-dispatch-srctools__init__py)
  - [8. Categorized Memory Tool (`src/tools/memory.py`)](#8-categorized-memory-tool-srctoolsmemorypy)
  - [9. Dynamic Tool Builder (`src/tools/tool_builder.py`)](#9-dynamic-tool-builder-srctoolstool_builderpy)
  - [10. CLI and Project File Manager (`src/tools/cli.py`)](#10-cli-and-project-file-manager-srctoolsclipy)
  - [11. Extended Calculator Tool (`src/tools/calculator.py`)](#11-extended-calculator-tool-srctoolscalculatorpy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Clone and Environment Setup](#clone-and-environment-setup)
  - [Environment Configuration](#environment-configuration)
  - [Dependency Installation](#dependency-installation)
  - [Database Migrations](#database-migrations)
- [Usage Examples](#usage-examples)
  - [1. Server Startup](#1-server-startup)
  - [2. Standard Chat with Token Metrics](#2-standard-chat-with-token-metrics)
  - [3. Autonomous Tool-Assisted Execution](#3-autonomous-tool-assisted-execution)
  - [4. Event Stream Chat](#4-event-stream-chat)
  - [5. Database Migration Commands](#5-database-migration-commands)
- [How It Works](#how-it-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)
  - [Milestone 1: Initial Architecture and Tool Loop](#milestone-1-initial-architecture-and-tool-loop)
  - [Milestone 2: System Prompt Externalization and Documentation](#milestone-2-system-prompt-externalization-and-documentation)
  - [Milestone 3: Settings Centralization and Schema Refactoring](#milestone-3-settings-centralization-and-schema-refactoring)
  - [Milestone 4: CLI File Management and Tool Integration](#milestone-4-cli-file-management-and-tool-integration)
  - [Milestone 5: Project Isolation and Dynamic Tool Builder](#milestone-5-project-isolation-and-dynamic-tool-builder)
  - [Milestone 6: Calculator Enhancements, Loop Safeguards, and Modular Segregation](#milestone-6-calculator-enhancements-loop-safeguards-and-modular-segregation)
  - [Milestone 7: Pydantic Settings Management and Web Search Tooling](#milestone-7-pydantic-settings-management-and-web-search-tooling)
  - [Milestone 8: Persistent Categorized Memory and Async Background Consolidation](#milestone-8-persistent-categorized-memory-and-async-background-consolidation)
  - [Milestone 9: Alembic Database Migrations and Prompt Version Tracking](#milestone-9-alembic-database-migrations-and-prompt-version-tracking)
  - [Milestone 10: Agent Engine Decoupling and Instruction Modularization](#milestone-10-agent-engine-decoupling-and-instruction-modularization)

## Project Overview

Building production-ready autonomous agents requires combining local language models with deterministic tools, bounded iteration loops, long-term memory consolidation, and robust schema migrations. This project demonstrates an agent backend capable of reasoning across turns, building and testing custom Python tools on the fly, storing user knowledge in structured markdown files, tracking prompt and configuration versions in an asynchronous SQL database, and enforcing security boundaries across file operations.

1. **Decoupled Execution Architecture**: Separates the HTTP presentation layer (`src/main.py`) from the core agent execution engine (`src/agent.py`), allowing background execution and isolated thread dispatching.
2. **Persistent Categorized Memory**: Maintains structured Markdown files in `memory/` and executes asynchronous consolidation passes after each conversation turn.
3. **Database Prompt Versioning**: Uses SQLModel, SQLAlchemy Async, and Alembic migrations to hash configurations and activate version records on application startup.
4. **Self-Extending Dynamic Tooling**: Allows the agent to write new Python tools, parse them with AST, verify them with automated tests, and dynamically load passed tools at runtime.
5. **Bounded Execution and Circuit Breakers**: Implements a 25-step execution limit with repeated call detection (`seen >= 3`) and consecutive error thresholds (`errors >= 4`).

## Key Features

- **Multi-Modal API Endpoints**:
  - Non-streaming chat endpoint (`/api/chat`) returning assistant responses and token usage metrics.
  - Autonomous multi-turn reasoning endpoint (`/api/tool`) returning answers, step traces, and execution telemetry.
  - Real-time token streaming (`/api/stream`) using FastAPI `StreamingResponse` over `text/event-stream`.
- **Decoupled Agent Engine (`run_agent`)**:
  - Bounded iteration loop up to 25 steps when tools are enabled.
  - Step telemetry capturing traces, step counts, total input/output tokens, and peak prompt tokens.
  - Built-in circuit breakers catching repetitive calls and consecutive error bursts.
  - Developer mode (`DEV_MODE`) toggling real-time console streaming of thinking blocks, content, and tool calls.
- **Persistent Long-Term Memory System**:
  - Topic-specific Markdown files (`user_profile.md`, `projects.md`, `topics.md`) with file locks and line bounds.
  - Thread-safe memory tool (`memory_tool`) for reading, creating, appending, and updating lines.
  - Asynchronous background consolidation (`memory_pass`) extracting personal info, preferences, and project updates.
  - Dynamic prompt injection incorporating the live memory index into system context.
- **Database & Prompt Version Management**:
  - Asynchronous database engine with `sqlite+aiosqlite` and Alembic migration scripts.
  - `PromptVersion` model maintaining active configuration hashes, system prompts, tool instructions, and model settings.
  - Single-active-version constraint enforced at the database level.
- **Sandboxed Workspace File Management (`cli_tool`)**:
  - Scoped project directories inside `artifacts/<project>`.
  - Path traversal validation, shell meta-character blocking, and binary allowlisting.
  - Line-level editing (`update_lines`) avoiding entire file rewrites.
- **Dynamic Tool Creation (`tool_builder`)**:
  - AST-based parameter extraction and automatic schema generation (`spec.json`).
  - Isolated subprocess test execution requiring a passed assertion suite (`.passed` marker).
  - Runtime dynamic discovery (`load_custom_tools`) combining pre-configured and dynamic tools.
- **Extended Calculation Engine (`calculator_tool`)**:
  - 17 binary operations (roots, arbitrary base logarithms, gcd, lcm, percentages, hypotenuse).
  - 11 unary operations (trigonometry in degrees, factorials, logarithms, absolute values).

## Core Components

### 1. Application API Server (`src/main.py`)

Serves as the HTTP routing layer. Manages application lifespan initialization, dispatches agent execution to worker threads, triggers background memory consolidation tasks, and streams token chunks.

| Endpoint | Implementation | Behavior |
| :--- | :--- | :--- |
| Lifespan Handler | `@asynccontextmanager async def lifespan(app)` | Executes database configuration hashing and active version resolution via `init()`. |
| `/api/chat` | `@app.post("/api/chat")` | Runs `run_agent(tool_use=False)` in a worker thread, spawns background `memory_pass`, and returns content and token metrics. |
| `/api/tool` | `@app.post("/api/tool")` | Runs `run_agent(tool_use=True)` in a worker thread, spawns background `memory_pass`, and returns content and token metrics. |
| `/api/stream` | `@app.post("/api/stream")` | Formats non-tool messages and returns `StreamingResponse` wrapping `stream_response`. |

```python
@app.post("/api/chat")
async def normal_chat(query: Query):
    run = await asyncio.to_thread(run_agent, query.input, query.model, False)
    asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)

@app.post("/api/tool")
async def tool_chat(query: Query):
    run = await asyncio.to_thread(run_agent, query.input, query.model, True)
    asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)
```

### 2. Autonomous Agent Engine (`src/agent.py`)

Contains the core reasoning loop. Coordinates context preparation, dynamic tool loading, Ollama chat invocations, function execution, step telemetry, and loop safeguards.

| Component | Implementation | Behavior |
| :--- | :--- | :--- |
| `MAX_STEPS` | `int = 25` | Maximum iterations permitted in a single tool-calling request. |
| `run_agent` | `def run_agent(...) -> dict` | Executes model interaction loop, dynamically imports tools, tracks tokens, checks circuit breakers, and returns structured run output. |

```python
def run_agent(user_input: str, model: str | None = None, tool_use: bool = False) -> dict:
    model = model or SETTINGS.DEFAULT_MODEL
    trim_chat()
    chat_history.append(Message(role="user", content=user_input))

    trace, breakers = [], []
    total_in = total_out = max_step_in = 0
    seen, errors = {}, 0
    answer, steps = "Stopped after too many steps.", 0

    for i in range(MAX_STEPS if tool_use else 1):
        steps = i + 1

        if tool_use:
            custom_tools, custom_funcs = load_custom_tools()
            all_tools = tools + custom_tools
            all_funcs = {**functions, **custom_funcs}
        else:
            all_tools, all_funcs = None, {}

        res: ChatResponse = client.chat(
            model=model,
            messages=build_messages(tool_use=tool_use),         
            tools=all_tools,
            think=SETTINGS.THINKING,
            options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
        )
        chat_history.append(res.message)
```

### 3. Centralized Configuration (`src/settings.py`)

Uses Pydantic `BaseSettings` to manage environment variables from `.env`, directory constants, model defaults, and cached Ollama clients.

| Setting | Implementation | Behavior |
| :--- | :--- | :--- |
| `DEV_MODE` | `bool = True` | Enables verbose terminal logging of thinking traces and tool calls. |
| `DATABASE_URL` | `str = "sqlite+aiosqlite:///./agent.db"` | Async SQLAlchemy connection string. |
| `DEFAULT_MODEL` | `str = "gemma4:12b"` | Default Ollama model name. |
| `MODEL_CTX_SZ` | `int = 8192` | Model context window size in tokens. |
| `OLLAMA_API_KEY` | `str` | API key used for authorized Ollama requests. |
| `OLLAMA_WEB_SEARCH` | `bool = False` | Flag enabling conditional registration of web search tools. |
| `SYSTEM_PROMPT` | `Field(default_factory=...)` | Loads personality and behavioral instructions from `system-prompt.txt`. |
| `TOOL_INSTRUCTIONS` | `Field(default_factory=...)` | Loads tool creation and execution rules from `tool-instructions.txt`. |
| `MEMORY_INSTRUCTIONS` | `Field(default_factory=...)` | Loads memory categorization and indexing rules from `memory-instructions.txt`. |

```python
class Settings(BaseSettings):
    DEV_MODE: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///./agent.db"

    DEFAULT_MODEL: str = "gemma4:12b"
    MODEL_CTX_SZ: int = 8192
    THINKING: bool = True

    OLLAMA_API_KEY: str
    OLLAMA_WEB_SEARCH: bool = False

    ARTIFACTS_FOLDER_NAME: str = "artifacts"
    MEMORY_FOLDER_NAME: str = "memory"
    AGENT_TOOLS_FOLDER_NAME: str = "agent-tools"

    SYSTEM_PROMPT: str = Field(default_factory=lambda: (BASE_DIR / "system-prompt.txt").read_text())
    TOOL_INSTRUCTIONS: str = Field(default_factory=lambda: (BASE_DIR / "tool-instructions.txt").read_text())
    MEMORY_INSTRUCTIONS: str = Field(default_factory=lambda: (BASE_DIR / "memory-instructions.txt").read_text())

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    @cached_property
    def OLLAMA_CLIENT(self) -> Client:
        return Client(headers={"Authorization": f"Bearer {self.OLLAMA_API_KEY}"})
```

### 4. Long-Term Memory & Message Assembly (`src/memory.py`)

Manages conversational message history, contextual prompt construction, and background consolidation passes.

| Function | Implementation | Behavior |
| :--- | :--- | :--- |
| `build_messages` | `def build_messages(tool_use: bool) -> list` | Dynamically stitches `SYSTEM_PROMPT`, `TOOL_INSTRUCTIONS`, `MEMORY_INSTRUCTIONS`, and `memory_index()` when `tool_use=True`; sanitizes tool metadata when `tool_use=False`. |
| `memory_pass` | `async def memory_pass(user_msg, assistant_msg)` | Autonomous background agent pass prompting the model with `memory_tool_spec` to extract and persist key facts to `memory/`. |

```python
def build_messages(tool_use: bool) -> list:
    if tool_use:
        system_text = f"""
            {SETTINGS.SYSTEM_PROMPT}\n\n
            {SETTINGS.TOOL_INSTRUCTIONS}\n\n
            {SETTINGS.MEMORY_INSTRUCTIONS}\n\n
            Your memory files:\n
            {memory_index()}
        """
    else:
        system_text = (
            f"{SETTINGS.SYSTEM_PROMPT}\n\n"
            "You have no tools in this mode. Do not claim to use tools, "
            "run code, or save or read memory."
        )

    rest = []
    for m in messages[1:]:
        d = m if isinstance(m, dict) else m.model_dump()
        if not tool_use:
            if d.get("role") == "tool":
                continue
            if d.get("tool_calls"):
                if not d.get("content"):
                    continue
                d = {**d, "tool_calls": None}
        rest.append(d)
    return [{"role": "system", "content": system_text}, *rest]
```

### 5. Database Layer & Prompt Versioning (`src/db/models.py`)

Defines SQLModel database entities and version activation logic.

| Entity / Function | Implementation | Behavior |
| :--- | :--- | :--- |
| `PromptVersion` | `SQLModel, table=True` | Stores prompt version records with unique configuration hashes and partial index enforcing a single active prompt. |
| `active` | `async def active(db, version_id)` | Deactivates existing active prompt versions and marks the specified version as active. |

```python
class PromptVersion(SQLModel, table=True):
    __tablename__ = "prompt_versions"
    __table_args__ = (
        Index(
            "one_active_prompt",
            "is_active",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active"),
        ),
    )

    id: int | None = Field(primary_key=True, default=None)
    model: str
    is_active: bool = Field(default=False)
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))
    tool_instructions: str = Field(sa_column=Column(Text, nullable=False))
    memory_instructions: str = Field(sa_column=Column(Text, nullable=False))
    config_hash: str = Field(sa_column=Column(Text, nullable=False))
    settings: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### 6. Utilities and Lifespan Initialization (`src/utils.py`)

Handles application initialization, token streaming, response mapping, and context truncation.

| Function | Implementation | Behavior |
| :--- | :--- | :--- |
| `init` | `async def init()` | Generates a SHA-256 hash of the current runtime configuration, ensures a `PromptVersion` record exists in the database, and activates it. |
| `stream_response` | `async def stream_response(...)` | Iterates over streaming chunks from Ollama, yields tokens, stores assistant message, and dispatches background memory pass. |
| `trim_chat` | `def trim_chat(max_messages=20)` | Preserves `messages[0]` and slices chat turns to keep the latest $N$ turns starting with a user turn. |
| `to_response` | `def to_response(run)` | Converts internal agent run dictionary to public API response schema. |

```python
async def init():
    cfg = build_config()
    h = config_hash(cfg)
    async with AsyncSession(engine) as db:
        result = await db.exec(select(PromptVersion).where(PromptVersion.config_hash == h))
        row = result.first()
        if not row:
            row = PromptVersion(**cfg, config_hash=h)
            db.add(row)
            await db.commit()
            await db.refresh(row)
        if not row.is_active:
            await active(db, row.id)
```

### 7. Tool Registry and Dynamic Dispatch (`src/tools/__init__.py`)

Assembles the base tool list and maps tool names to Python callables.

| Tool | Registration Name | Behavior |
| :--- | :--- | :--- |
| `calculator_tool` | `calculator_tool` | Evaluates arithmetic, logarithmic, and trigonometric calculations. |
| `tool_builder` | `tool_builder` | Dynamically creates, tests, reads, updates, and runs custom Python tools. |
| `memory_tool` | `memory_tool` | Manages persistent categorized markdown files in `memory/`. |
| `cli_tool` | `cli_tool` | Executes sandboxed project commands and manages files within `artifacts/`. |
| `web_search` | `web_search` | Conditionally registered when `OLLAMA_WEB_SEARCH=True` to query search results. |
| `web_fetch` | `web_fetch` | Conditionally registered when `OLLAMA_WEB_SEARCH=True` to extract full webpage content. |

```python
from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool
from .cli import cli_tool
from .memory import memory_tool
from src.settings import SETTINGS
from .tool_builder import tool_builder

tools: List[Tool] = []
functions: Dict[str, function] = {}

functions["calculator_tool"] = calculator_tool
tools.append(calculator_tool_spec)

functions["tool_builder"] = tool_builder
tools.append(tool_builder_spec)

functions["memory_tool"] = memory_tool
tools.append(memory_tool_spec)

functions["cli_tool"] = cli_tool
tools.append(cli_tool_spec)
```

### 8. Categorized Memory Tool (`src/tools/memory.py`)

Implements thread-safe file operations over Markdown files stored in `memory/`.

| Action | Implementation | Behavior |
| :--- | :--- | :--- |
| `list` | `memory_tool(action="list")` | Returns formatted overview of all memory files and their first line descriptions. |
| `read` | `memory_tool(action="read", ...)` | Returns numbered lines from a memory file with optional line range selection. |
| `create` | `memory_tool(action="create", ...)` | Creates a new category file under `memory/` (maximum 50 files). |
| `append` | `memory_tool(action="append", ...)` | Appends new lines to an existing category file (maximum 200 lines per file). |
| `update_lines` | `memory_tool(action="update_lines", ...)` | Replaces specific 1-indexed line numbers in place. |
| `delete_lines` | `memory_tool(action="delete_lines", ...)` | Deletes a range of lines from a memory file. |
| `delete_file` | `memory_tool(action="delete_file", ...)` | Deletes a memory file from disk. |

```python
def memory_tool(
    action: str,
    file: str = "",
    content: str = "",
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    try:
        with _LOCK:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)

            if action == "list":
                return memory_index()

            if not file:
                return "Error: 'file' is required. Call list to see existing files"
            p = _path(file)
            ...
```

### 9. Dynamic Tool Builder (`src/tools/tool_builder.py`)

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

### 10. CLI and Project File Manager (`src/tools/cli.py`)

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

### 11. Extended Calculator Tool (`src/tools/calculator.py`)

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
├── .env.sample                       # Template for environment configuration
├── Makefile                          # Development and migration command shortcuts
├── README.md                         # Project documentation
├── alembic.ini                       # Alembic migration configuration
├── memory-instructions.txt           # Rules for memory categorization and note formats
├── requirements.txt                  # Python package dependencies
├── system-prompt.txt                 # Agent personality and core conversational guidelines
├── tool-instructions.txt             # Rules for tool selection, construction, and file edits
├── agent-tools/                      # Storage directory for custom-built tools
│   ├── days_to_weeks/                # Custom tool: converts days to weeks and remaining days
│   │   ├── description.txt           # Tool description used in LLM tool definition
│   │   ├── spec.json                 # JSON schema of tool parameters
│   │   ├── test_tool.py              # Test suite for days_to_weeks
│   │   └── tool.py                   # Python implementation of days_to_weeks
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
├── memory/                           # Long-term agent memory category files
│   ├── projects.md                   # Tracks ongoing user projects and stack details
│   ├── topics.md                     # Records explored topics and subjects
│   └── user_profile.md               # User preferences, role, and learning topics
├── migrations/                       # Database migration scripts
│   ├── README                        # Alembic migration readme
│   ├── env.py                        # Migration runtime environment
│   ├── script.py.mako                # Revision template file
│   └── versions/                     # Revision migration versions
│       ├── 316a13cb13e0_init.py      # Initial schema migration
│       └── f76e34ad29da_.py          # Prompt versions schema enhancement
└── src/
    ├── __init__.py                   # Root package marker
    ├── agent.py                      # Core agent reasoning loop, tool execution, and telemetry
    ├── main.py                       # FastAPI application and route endpoints
    ├── memory.py                     # Message context builder and background memory consolidation
    ├── schemas.py                    # Pydantic input and output request schemas
    ├── settings.py                   # Pydantic BaseSettings configuration and client provider
    ├── utils.py                      # Lifespan database init, streaming, and history trimming
    ├── db/
    │   ├── __init__.py               # Database package marker
    │   ├── main.py                   # Async database engine and session generator
    │   └── models.py                 # SQLModel prompt version models and activation helpers
    └── tools/
        ├── __init__.py               # Tool registry importing static tools, web search, and schemas
        ├── calculator.py             # Extended arithmetic, unary, and binary math tool
        ├── cli.py                    # Project-scoped CLI runner and safe file manager
        ├── memory.py                 # Thread-safe markdown category memory file manager
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

### Environment Configuration

Create a local `.env` configuration file from the template:

```bash
cp .env.sample .env
```

Configure your environment values in `.env`:

```env
# OLLAMA CONFIGURATIONS
OLLAMA_API_KEY="your-api-key-here"

# MODEL CONFIGURATIONS
DEFAULT_MODEL="gemma4:12b"
THINKING_MODEL=True
MODEL_CTX_SZ=8192

# TOOLS CONFIGURATIONS
ARTIFACTS_FOLDER_NAME="artifacts"
AGENT_TOOLS_FOLDER_NAME="agent-tools"
MEMORY_FOLDER_NAME="memory"
```

### Dependency Installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Database Migrations

Apply database migrations to set up the `prompt_versions` table in `agent.db`:

```bash
make migrate
```

Or using Alembic directly:

```bash
alembic upgrade head
```

## Usage Examples

### 1. Server Startup

Run the development server via Make or Uvicorn:

```bash
make dev
```

Or using Uvicorn directly:

```bash
uvicorn src.agent.main:app --reload --reload-dir src
```

### 2. Standard Chat with Token Metrics

Query `/api/chat` to receive assistant replies and token consumption metrics:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is the difference between synchronous and asynchronous execution?"
  }'
```

### 3. Autonomous Tool-Assisted Execution

Query `/api/tool` to execute multi-step requests involving math calculations, persistent memory updates, project scaffolding, or dynamic tool construction:

```bash
curl -X POST http://localhost:8000/api/tool \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Remember that I am learning agentic AI architecture with FastAPI, and convert 45 days into weeks."
  }'
```

### 4. Event Stream Chat

Stream token chunks directly from `/api/stream`:

```bash
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "input": "Explain the role of long-term memory in AI agents."
  }'
```

### 5. Database Migration Commands

Manage database schema revisions using Make:

```bash
# Create a new migration revision
make migration

# Apply all pending migrations
make migrate

# Rollback the last migration
make rollback
```

## How It Works

1. **Lifespan Startup and Prompt Versioning**: When FastAPI launches, `init()` builds a configuration dictionary containing the model name, system prompt, tool instructions, memory instructions, thinking flag, and context size. A SHA-256 hash is computed. If the configuration does not exist in the database, a new `PromptVersion` record is inserted and marked as active.
   ```python
   async def init():
       cfg = build_config()
       h = config_hash(cfg)
       async with AsyncSession(engine) as db:
           result = await db.exec(select(PromptVersion).where(PromptVersion.config_hash == h))
           row = result.first()
           if not row:
               row = PromptVersion(**cfg, config_hash=h)
               db.add(row)
               await db.commit()
               await db.refresh(row)
           if not row.is_active:
               await active(db, row.id)
   ```

2. **Context Assembly and Turn Trimming**: When a request arrives, `trim_chat` retains the system message at index `0` and slices the tail to keep only the latest turns. `build_messages` injects the externalized system prompt, tool instructions, memory instructions, and the current memory index into system context.
   ```python
   def build_messages(tool_use: bool) -> list:
       if tool_use:
           system_text = f"""
               {SETTINGS.SYSTEM_PROMPT}\n\n
               {SETTINGS.TOOL_INSTRUCTIONS}\n\n
               {SETTINGS.MEMORY_INSTRUCTIONS}\n\n
               Your memory files:\n
               {memory_index()}
           """
   ```

3. **Autonomous Agent Execution Loop**: `run_agent` in `src/agent.py` loops up to 25 steps when `tool_use=True`. At each step, dynamic custom tools are discovered via `load_custom_tools` and merged with static tools (`calculator_tool`, `cli_tool`, `memory_tool`, `tool_builder`). The model is invoked with the configured context size.
   ```python
   res: ChatResponse = client.chat(
       model=model,
       messages=build_messages(tool_use=tool_use),
       tools=all_tools,
       think=SETTINGS.THINKING,
       options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
   )
   ```

4. **Circuit Breakers and Tool Dispatch**: Function calls are dispatched to their corresponding Python handlers. Output is inspected by loop breakers:
   - If 4 consecutive tool executions return errors, execution halts.
   - If an identical tool call and result occur 3 times, a circuit breaker stops the loop.
   ```python
   errors = errors + 1 if result.startswith("Error") else 0
   if errors >= 4:
       result += "\nToo many errors in a row. Stop and tell the user what is failing."
       breakers.append("errors")

   key = (name, json.dumps(args, sort_keys=True, default=str), result)
   seen[key] = seen.get(key, 0) + 1
   if seen[key] >= 3:
       result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."
       breakers.append("repeat")
   ```

5. **Asynchronous Memory Consolidation**: Once the primary interaction concludes, the server dispatches `asyncio.create_task(memory_pass(query.input, run["answer"]))`. In the background, an evaluation agent inspects whether the exchange contains new user details, preferences, or project updates, calling `memory_tool` to update files in `memory/`.
   ```python
   @app.post("/api/tool")
   async def tool_chat(query: Query):
       run = await asyncio.to_thread(run_agent, query.input, query.model, True)
       asyncio.create_task(memory_pass(query.input, run["answer"]))
       return to_response(run)
   ```

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

### Milestone 7: Pydantic Settings Management and Web Search Tooling

- `036acc2`: feat: refactor settings management and enhance tool functionalities with new web search and fetch capabilities
  - Refactored `src/settings.py` to use `pydantic_settings.BaseSettings` with environment variable loading and cached client properties.
  - Added `.env.sample` defining environment variables including `OLLAMA_API_KEY`.
  - Integrated `web_search` and `web_fetch` from `SETTINGS.OLLAMA_CLIENT` into `src/tools/__init__.py`.
  - Added terminal logging for thinking traces and tool calls in `src/main.py`.
  - Added `ollama==0.6.3` into `requirements.txt`.

### Milestone 8: Persistent Categorized Memory and Async Background Consolidation

- `62d27b2`: feat: enhance memory management with summary functionality and integrate async processing in chat endpoints
  - Added structured memory summarization and async background processing for conversation turns.
- `fec8fd0`: feat: implement days to weeks conversion tool with memory management and API integration
  - Implemented `memory_tool` in `src/tools/memory.py` managing categorized markdown files in `memory/`.
  - Added `days_to_weeks` custom tool under `agent-tools/`.
- `b1e4ffb`: feat: add memory folder configuration and enable conditional web search tool loading
  - Added `MEMORY_FOLDER_NAME` in `src/settings.py` and `.env.sample`.
  - Added `OLLAMA_WEB_SEARCH` flag in settings to conditionally load web search and web fetch tools.

### Milestone 9: Alembic Database Migrations and Prompt Version Tracking

- `1544dee`: feat: implement database migration setup with Alembic and async support
  - Integrated Alembic migration environment (`migrations/`, `alembic.ini`).
  - Added async database setup in `src/db/main.py` using SQLModel and SQLAlchemy async engine.
  - Created initial migration revision `316a13cb13e0_init.py`.
- `710b8f6`: feat: add migration for prompt_versions table and enhance model with new fields
  - Defined `PromptVersion` model in `src/db/models.py` with unique active prompt index.
  - Added migration script `f76e34ad29da_.py` enhancing `prompt_versions` table.
  - Added `init()` in `src/utils.py` to hash configurations and activate matching prompt versions on application startup.

### Milestone 10: Agent Engine Decoupling and Instruction Modularization

- `b275cbf`: feat: add memory management instructions and refactor agent functionality with tool integration
  - Decoupled agent execution logic from `src/main.py` into dedicated engine `src/agent.py` (`run_agent`).
  - Modularized prompt instructions into `system-prompt.txt`, `tool-instructions.txt`, and `memory-instructions.txt`.
  - Integrated dynamic memory index injection and dual-mode message filtering in `src/memory.py` (`build_messages`).
  - Added background consolidation task (`memory_pass`) after user interactions.
