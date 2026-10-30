# ai-agent

ai-agent is an asynchronous Python service built with FastAPI and the Ollama SDK for running local large language model interactions and tool calling workflows from scratch. The service exposes HTTP endpoints for non-streaming conversational turns, token-level chunk streaming, and autonomous tool calling loops where model-generated function invocations are dispatched to Python handlers and fed back into context. It incorporates conversation state management, sliding-window turn trimming, and strict system prompt guidelines.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
  - [1. Application Server (`src/main.py`)](#1-application-server-srcmainpy)
  - [2. Tool Registry (`src/tools/__init__.py`)](#2-tool-registry-srctools__init__py)
  - [3. Arithmetic Calculator Tool (`src/tools/calculator.py`)](#3-arithmetic-calculator-tool-srctoolscalculatorpy)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Clone and Environment Setup](#clone-and-environment-setup)
  - [Dependency Installation](#dependency-installation)
- [Usage Examples](#usage-examples)
  - [1. Server Startup](#1-server-startup)
  - [2. Standard Chat Request](#2-standard-chat-request)
  - [3. Tool-Assisted Chat Request](#3-tool-assisted-chat-request)
  - [4. Streaming Chat Request](#4-streaming-chat-request)
- [How It Works](#how-it-works)
- [Project Roadmap & Commit History](#project-roadmap--commit-history)
  - [Milestone 1: Initial Architecture and Tool Loop](#milestone-1-initial-architecture-and-tool-loop)

## Project Overview

Learning to build AI agents requires managing the interaction cycle between an LLM and local code: maintaining conversational context, parsing model-requested tool calls, executing deterministic local functions, and returning execution output back to the model.

1. **Asynchronous Ollama Integration**: Communicates with local Ollama runtime models using `ollama.AsyncClient` for chat completions and streaming.
2. **Conversation History Trimming**: Preserves the primary system instructions while keeping only the latest $N$ conversation turns via `trim_chat`.
3. **Iterative Tool Calling**: Executes multi-step model function requests within a `while True` loop until a final text response is synthesized.
4. **FastAPI Transport Layer**: Provides standardized JSON and streaming HTTP endpoints for frontend and client consumption.

## Key Features

- **Multi-Modal Interaction Modes**:
  - Direct request and response chat via `/api/chat`.
  - HTTP token streaming responses via `/api/stream` using FastAPI `StreamingResponse`.
  - Autonomous multi-turn tool calling through `/api/tool`.
- **Tool Dispatch System**:
  - Declarative function schemas using `ollama.Tool` specifications.
  - Runtime execution dispatch using an internal dictionary registry (`functions`).
  - Reasoning trace visibility enabled through the `think=True` Ollama parameter.
- **Arithmetic Tooling**:
  - Isolated mathematical execution supporting addition, subtraction, multiplication, division, integer division, modulo, and exponentiation.
  - Safe error handling for zero division and invalid operators without raising uncaught server exceptions.
- **Stateful Memory Management**:
  - Global conversational memory storing structured `Message` objects.
  - Boundary enforcement limiting retained turns to prevent unbounded context growth.

## Core Components

### 1. Application Server (`src/main.py`)

Configures the FastAPI application instance, maintains global conversation state, and defines API endpoints. Implements `stream_response` for token yielding and `trim_chat` for keeping conversation length within limits.

| Name | Implementation | Behavior |
| :--- | :--- | :--- |
| `SYSTEM_PROMPT` | `str` | Instructs the model to answer concisely, avoid guessing, avoid emojis, and decline unsafe requests. |
| `Query` | `BaseModel` | Request schema accepting `model: str` and `input: str`. |
| `Message` | `BaseModel` | Message schema accepting `role: str` and `content: str`. |
| `stream_response` | `async def stream_response(...)` | Asynchronous generator that streams chat chunks from Ollama and stores assistant response in message history. |
| `trim_chat` | `def trim_chat(...)` | Preserves `msgs[0]` (system prompt) and slices remaining messages to `max_turns`. |
| `/api/chat` | `@app.post('/api/chat')` | Appends user message, calls `client.chat`, appends response, and returns assistant text. |
| `/api/tool` | `@app.post('/api/tool')` | Loops over model tool calls, invokes functions, appends tool outputs, and returns final text. |
| `/api/stream` | `@app.post('/api/stream')` | Returns a `StreamingResponse` wrapping `stream_response`. |

```python
@app.post('/api/tool')
async def tool_chat(query: Query):
    client = AsyncClient()
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))

    while True:
        formatted_msgs = []
        for m in messages:
            formatted_msgs.append(m if isinstance(m, dict) else m.model_dump())
        res: ChatResponse = await client.chat(
            model=query.model, 
            messages=formatted_msgs,
            tools=tools,
            think=True
        )
        messages.append(res.message)

        if res.message.tool_calls:
            for tc in res.message.tool_calls:
                name, args = tc.function.name, tc.function.arguments
                if name in functions:
                    print(f"Calling {name}({args})")
                    res = functions[name](**args)
                    print(f"Tool Result: {str(res)}")
                    messages.append({"role": "tool","content": str(res)})
            continue
        else:
            break
    return res.message.content
```

### 2. Tool Registry (`src/tools/__init__.py`)

Defines the exported tool lists and registers callable Python functions mapped to schemas accepted by Ollama.

| Name | Implementation | Behavior |
| :--- | :--- | :--- |
| `tools` | `List[Tool]` | List of Ollama `Tool` objects passed to the chat client. |
| `functions` | `Dict[str, function]` | Dictionary mapping tool names to callable Python functions. |
| `calculator_tool` definition | `Tool(type='function', function=...)` | Exposes calculator metadata and parameters (`op1`, `op2`, `operator`) to the model. |

```python
from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool

tools: List[Tool] = []
functions: Dict[str, function] = {}

# Calculator Tool
functions['calculator_tool'] = calculator_tool
tools.append(Tool(
    type='function',
    function=Tool.Function(
        name="calculator_tool",
        description="Perform basic arithmetic on two numbers. After final operations explain the answer naturally",
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "op1": {
                    "type": "float",
                    "description": "First Operand"
                },
                "op2": {
                    "type": "float",
                    "description": "Second Operand"
                },
                "operator": {
                    "type": "string",
                    "enum": list(_OPS.keys()),
                    "description": "Operator"
                },
            },
            required=["op1", "op2", "operator"]
        )
    )
))
```

### 3. Arithmetic Calculator Tool (`src/tools/calculator.py`)

Implements basic arithmetic execution using standard library `operator` mappings. Validates operands and operators before executing calculations.

| Name | Implementation | Behavior |
| :--- | :--- | :--- |
| `_OPS` | `dict` | Maps `+`, `-`, `*`, `/`, `//`, `%`, and `**` to `operator` functions. |
| `calculator_tool` | `def calculator_tool(...)` | Executes calculation; returns result string or error messages for unsupported operators and division by zero. |

```python
import operator

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}

def calculator_tool(op1: float, op2: float, operator: str) -> str:
    if operator not in _OPS:
        return "Error: unsupported operator"
    try:
        return str(_OPS[operator](op1, op2))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: invalid input"
```

## Project Structure

```text
ai-agent/
├── .gitignore             # Git ignore patterns for Python, virtual environments, and cache files
├── readme-generator.md    # Reference specification and prompt for generating project README
├── requirements.txt       # Python package dependencies
├── system-prompt.py       # Empty script file
└── src/
    ├── __init__.py        # Root package marker file
    ├── main.py            # FastAPI server, conversation memory, and API route handlers
    └── tools/
        ├── __init__.py    # Tool schema declarations and function dispatch registry
        └── calculator.py  # Calculator execution logic and arithmetic operator mappings
```

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Ollama service running locally with a tool-capable model (for example, `llama3.2` or `qwen2.5`)

Operating system installation commands for Ollama:

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

Start the Ollama daemon and pull a model:

```bash
ollama pull llama3.2
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

Run the FastAPI application with Uvicorn from the project root:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Standard Chat Request

Send a request to `/api/chat` for standard conversational replies:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "input": "Explain the difference between synchronous and asynchronous execution."
  }'
```

### 3. Tool-Assisted Chat Request

Send a request to `/api/tool` requiring calculations. The model recognizes the tool, generates tool calls, receives results, and replies:

```bash
curl -X POST http://localhost:8000/api/tool \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "input": "What is 482 multiplied by 19?"
  }'
```

### 4. Streaming Chat Request

Send a request to `/api/stream` to receive raw token chunks:

```bash
curl -X POST http://localhost:8000/api/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "llama3.2",
    "input": "Give me three best practices for designing REST APIs."
  }'
```

## How It Works

1. **Context Initialization and Turn Trimming**: The global `messages` array begins with `SYSTEM_PROMPT`. When a request arrives, `trim_chat` retains the system message at index `0` and slices the tail to keep only the latest turns.
   ```python
   def trim_chat(msgs: List[Message], max_turns: int = 20):
       system_prompt = msgs[0]
       chat_turns = msgs[1:]
       max_messages = max_turns

       if len(chat_turns) > max_messages:
           chat_turns = chat_turns[-max_messages:]
       return [system_prompt] + chat_turns
   ```

2. **Model Chat with Registered Tools**: In `/api/tool`, formatted messages and tool specifications are forwarded to `client.chat` with `think=True`.
   ```python
   res: ChatResponse = await client.chat(
       model=query.model, 
       messages=formatted_msgs,
       tools=tools,
       think=True
   )
   messages.append(res.message)
   ```

3. **Autonomous Tool Call Dispatch**: If `res.message.tool_calls` contains calls, the server iterates through them, locates the matching function in `functions`, executes it with arguments, and appends the result to `messages` under role `tool`.
   ```python
   if res.message.tool_calls:
       for tc in res.message.tool_calls:
           name, args = tc.function.name, tc.function.arguments
           if name in functions:
               print(f"Calling {name}({args})")
               res = functions[name](**args)
               print(f"Tool Result: {str(res)}")
               messages.append({"role": "tool","content": str(res)})
       continue
   else:
       break
   ```

4. **Completion Output**: When no further tool calls are generated, the loop terminates and `res.message.content` is returned to the caller.

## Project Roadmap & Commit History

### Milestone 1: Initial Architecture and Tool Loop

- `5271013`: Add initial project structure with core functionality and dependencies
  - Added `.gitignore` to exclude build artifacts, caches, and environment directories.
  - Added `readme-generator.md` containing prompt requirements for documentation.
  - Added project dependencies in `requirements.txt`.
  - Implemented application server and routes (`/api/chat`, `/api/tool`, `/api/stream`) in `src/main.py`.
  - Implemented tool schema declarations in `src/tools/__init__.py`.
  - Implemented arithmetic operations in `src/tools/calculator.py`.

