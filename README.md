# AI Agent (FastAPI + LangChain + SQLModel + Alembic + MCP + RAG)

A streamlined, production-ready AI agent service scaffolded with **FastAPI**, **LangChain** (via `ChatOllama`), **SQLModel**, **Alembic**, **Model Context Protocol (MCP)**, and **RAG** vector search.

## Features

- **LangChain Integration**: Leverages `ChatOllama` and standard LangChain messages (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) for clean model interactions.
- **Dynamic MCP Server Integration**: Plug-and-play Model Context Protocol (MCP) server support loaded from a standardized `mcp_servers.json` configuration file.
- **RAG Document Ingestion**: Vector storage with chunking, embedding, and retriever-as-a-tool support.
- **Persistent Session Memory & Checkpointers**: Conversation history is persisted in SQLite using `langgraph-checkpoint-sqlite`. Sessions resume automatically when supplying a `session_id` (or auto-assign a UUID).
- **Auto-Summarizing Conversation History**: Tracks active sessions in `conversation_sessions`, maintaining an auto-updated topic summary column, message counts, and timestamps.
- **Custom Context Middlewares**:
  - Prunes old tool call results from previous turns into short placeholders to free context tokens.
  - Summarizes and compresses earlier conversation turns when context overflows.
- **Centralized API Endpoints**:
  - `POST /chat`: Standard JSON response with multi-step tool execution.
  - `POST /chat/stream`: Server-Sent Events (SSE) token streaming with `X-Session-ID` header.
  - `GET /conversations`: List all conversation sessions with `session_id`, `summary`, and `last_message_at`.
  - `GET /conversations/{session_id}`: Retrieve a conversation's full message history.
  - `DELETE /conversations/{session_id}`: Remove a conversation session.
  - `POST /documents`: Unified RAG document ingestion endpoint (JSON or file uploads).
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
│   │       ├── chat.py          # /chat, /chat/stream
│   │       ├── conversations.py # /conversations (list, detail, delete)
│   │       └── documents.py     # /documents (RAG ingest)
│   ├── agent/
│   │   ├── llm.py           # ChatOllama setup
│   │   ├── builder.py       # Assembles agent with tools and tool-calling loop
│   │   ├── memory.py        # SQLite checkpointer & session persistence
│   │   ├── middleware.py    # Tool pruning & context summarization middlewares
│   │   └── prompts.py       # System and chat prompt templates
│   ├── models/
│   │   └── conversation.py  # SQLModel for conversation sessions & topic summary
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

## Document Management & RAG Knowledge Base (ChromaDB)

Persistent vector search is powered by **ChromaDB** (`langchain-chroma`) persisting embeddings under `data/chroma/`. Every ingested document receives or is assigned a unique `doc_id`.

### 1. Ingest Documents (`POST /documents`)

The endpoint accepts **both** raw JSON text and multipart file uploads in a single route:

#### Option 1: Raw Text via JSON
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "LangChain enables modular LLM architectures.", "doc_id": "langchain-guide", "metadata": {"source": "manual"}}'
```

#### Option 2: File Upload (PDF, Word DOCX, Markdown, Text, Code)
```bash
# Upload a PDF, Word document (.docx), Markdown or text file
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_report.pdf" \
  -F "doc_id=quarterly-report-2026" \
  -F 'metadata={"category": "finance"}'
```

*Response:*
```json
{
  "status": "success",
  "doc_id": "quarterly-report-2026",
  "chunks_count": 8,
  "message": "Successfully ingested 8 chunk(s) from 'sample_report.pdf' into knowledge base."
}
```

### 2. List All Uploaded Documents (`GET /documents`)
```bash
curl http://localhost:8000/documents
```
*Response:*
```json
[
  {
    "doc_id": "quarterly-report-2026",
    "filename": "sample_report.pdf",
    "source": "sample_report.pdf",
    "chunks_count": 8,
    "created_at": "2026-11-14T18:30:00Z"
  }
]
```

### 3. Delete Document (`DELETE /documents/{doc_id}`)
Deletes the document entirely and cleanly from ChromaDB (all chunk embeddings) and removes the registry entry:
```bash
curl -X DELETE http://localhost:8000/documents/quarterly-report-2026
```
*Response:*
```json
{
  "status": "success",
  "doc_id": "quarterly-report-2026",
  "message": "Successfully deleted document 'quarterly-report-2026' and all its chunks from vector database."
}
```

---

## Listing & Managing Conversations

### List all active conversations:
```bash
curl http://localhost:8000/conversations
```
*Response:*
```json
[
  {
    "session_id": "session-101",
    "summary": "Docker container configuration and networking",
    "last_message_at": "2026-11-14T12:30:00Z",
    "created_at": "2026-11-14T12:00:00Z",
    "message_count": 4
  }
]
```

### Inspect conversation history:
```bash
curl http://localhost:8000/conversations/session-101
```

### Delete a conversation:
```bash
curl -X DELETE http://localhost:8000/conversations/session-101
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

### 2. Database Migrations & Management
```bash
# Apply pending migrations
make migrate

# Clean app.db and recreate all tables and migrations from scratch
make db-reset

# Clean app.db and re-apply existing migrations
make db-clean
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
