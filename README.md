# AI Agent (FastAPI + LangChain + SQLModel + Alembic + MCP + RAG)

A streamlined, production-ready AI agent service scaffolded with **FastAPI**, **LangChain** (via `ChatOllama`), **SQLModel**, **Alembic**, **Model Context Protocol (MCP)**, and **RAG** vector search.

## Features

- **LangChain Integration**: Leverages `ChatOllama` and standard LangChain messages (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) for clean model interactions.
- **Dynamic MCP Server Integration**: Plug-and-play Model Context Protocol (MCP) server support loaded from a standardized `mcp_servers.json` configuration file.
- **RAG Document Ingestion**: Vector storage with chunking, embedding, and retriever-as-a-tool support.
- **Centralized API Endpoints**:
  - `POST /chat`: Standard JSON response with multi-step tool execution.
  - `POST /chat/stream`: Server-Sent Events (SSE) token streaming.
  - `POST /documents`: RAG document ingestion endpoint.
  - `GET /health`: Service health check reporting active feature flags.
- **SQLModel Persistence**: Async SQLite / PostgreSQL session handling.
- **Alembic Migrations**: Fully configured async migration pipeline (`alembic upgrade head`) wired to SQLModel metadata.

---

## Project Structure

```
├── Makefile                 # Development, migration, and testing workflows
├── alembic.ini              # Alembic migration configuration
├── pyproject.toml / requirements.txt
├── mcp_servers.json         # Standardized MCP servers configuration
├── .env.sample              # Environment variable template
├── .gitignore
├── README.md
│
├── app/
│   ├── main.py              # FastAPI application factory and lifespan
│   ├── config.py            # Typed settings and environment configuration
│   ├── db.py                # Async database engine and session generator
│   ├── api/
│   │   ├── deps.py          # Shared dependencies (agent, vector store, session)
│   │   └── routes/
│   │       ├── chat.py      # /chat, /chat/stream
│   │       └── documents.py # /documents (RAG ingest)
│   ├── agent/
│   │   ├── llm.py           # ChatOllama setup
│   │   ├── builder.py       # Assembles agent with tools and tool-calling loop
│   │   └── prompts.py       # System and chat prompt templates
│   ├── tools/
│   │   ├── registry.py      # Aggregates local, RAG, and MCP tools
│   │   ├── local.py         # Plain @tool functions
│   │   ├── mcp.py           # Standardized MCP server loader
│   │   └── rag.py           # Retriever wrapped as a tool
│   ├── rag/
│   │   ├── store.py         # Vector store initialization (InMemory/Chroma/FAISS)
│   │   ├── ingest.py        # Load, chunk, embed
│   │   └── embeddings.py    # OllamaEmbeddings
│   └── schemas/
│       ├── chat.py          # Chat & Document request/response models
│       └── mcp.py           # Standardized MCP configuration schema
│
├── migrations/              # Alembic migration scripts
│   ├── env.py
│   └── versions/
│
├── tests/                   # Automated unit test suite
│   ├── test_api.py          # API route tests (/health, /chat, /documents)
│   ├── test_db.py           # SQLModel database operations tests
│   ├── test_agent.py        # Agent runner, local tools, and MCP schema tests
│   └── test_rag.py          # RAG chunking and vector retrieval tests
│
└── data/                    # Runtime storage (gitignored)
    └── app.db               # SQLite database
```

---

## Configuring MCP Servers (`mcp_servers.json`)

MCP servers are defined in `mcp_servers.json` in the standard multi-platform format (identical to Claude Desktop, Cursor, and VS Code):

```json
{
  "mcpServers": {
    "git": {
      "command": "python",
      "args": ["-m", "mcp_server_git", "--repository", "."],
      "env": {
        "GIT_AUTHOR_NAME": "SankalpPimpalkar",
        "GIT_AUTHOR_EMAIL": "sankalp.pimpalkar.dev@gmail.com"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
    },
    "remote_sse_service": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

### Schema Features:
- **`stdio` (local subprocess)**: Provide `"command"` and optional `"args"`, `"env"`, `"cwd"`.
- **`sse` (HTTP SSE endpoint)**: Provide `"url"` and optional `"headers"`.
- **`disabled: true`**: Temporarily disable any server without deleting its config.
- **Environment expansion**: Automatically resolves `$VAR` or `${VAR}` within `args` and `env`.

Enable MCP tools in `.env`:
```env
ENABLE_TOOLS=True
ENABLE_MCP=True
```

---

## Quickstart

### 1. Setup
```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
make install

# Configure environment
cp .env.sample .env
```

### 2. Database Migrations
```bash
make migrate
```

### 3. Run the Application
```bash
# Start development server with auto-reload
make dev

# Or run on production host:port
make start
```
API Documentation: `http://localhost:8000/docs`

---

## Running Tests
```bash
make test
make test-verbose
```
