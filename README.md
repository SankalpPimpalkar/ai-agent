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
  - `GET /health`: Service health check reporting active feature flags.
  - `GET /settings` & `POST /settings`: Centralized configuration endpoint for UI and clients to view and update system prompt, user instructions, sampling temperature, max iterations, and thinking mode with strict fishy-instruction security validation.
  - `POST /chat`: Standard JSON response with multi-step tool execution and configurable `thinking` mode.
  - `POST /chat/stream`: Server-Sent Events (SSE) token streaming with `X-Session-ID` header and configurable `thinking` mode.
  - `GET /conversations`: List all conversation sessions with `session_id`, `summary`, and timestamps.
  - `GET /conversations/{session_id}`: Retrieve a conversation's full message history.
  - `DELETE /conversations/{session_id}`: Remove a conversation session.
  - `POST /documents`: Unified RAG document ingestion endpoint (JSON raw text or multipart file uploads).
  - `GET /documents`: List all ingested documents and chunk counts in the vector knowledge base.
  - `GET /documents/chunks`: Search and fetch matching chunks via query parameters.
  - `POST /documents/query`: Search and fetch matching chunks via JSON body.
  - `DELETE /documents/{doc_id}`: Cleanly delete a document and its embeddings from the vector database.
  - `/evals/*`: Dedicated REST endpoints to manage isolated evaluation documents, run benchmarks, and inspect results.
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

## API Reference & Endpoints

All endpoints support JSON and standard RESTful HTTP semantics. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### 1. Health Check

#### `GET /health`
Returns service status, application name, and active feature flags.

- **Request**: None
- **cURL Example**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Response** (`200 OK`):
  ```json
  {
    "status": "ok",
    "app_name": "ai-agent",
    "rag_enabled": true,
    "tools_enabled": true,
    "mcp_enabled": true
  }
  ```

---

### 2. Settings & Configuration Endpoints (`/settings`)

Designed for UI layers, administrators, and clients to inspect and dynamically configure agent behavior, user instructions, sampling temperature, max iterations, and thinking mode in a dedicated endpoint. The core system prompt is internal and not exposed over the API (it can be updated locally via CLI tools like `cat` / `nano` editing `system-prompt.txt`).

#### `GET /settings`
Retrieves current runtime configuration, global user instructions, and feature flags.

- **Request**: None
- **cURL Example**:
  ```bash
  curl http://localhost:8000/settings
  ```
- **Response** (`200 OK`, `SettingsResponse`):
  ```json
  {
    "app_name": "ai-agent",
    "model": "qwen3.5:9b",
    "temperature": 0.7,
    "max_iterations": 10,
    "thinking": false,
    "instructions": "Be concise and format all answers in bullet points.",
    "rag_enabled": true,
    "tools_enabled": true,
    "mcp_enabled": true,
    "persistent_memory_enabled": true
  }
  ```

#### `POST /settings` (or `PUT /settings`, `PATCH /settings`)
Updates runtime configuration and user instructions. The system prompt is an abstract, immutable baseline loaded from `system-prompt.txt` / configuration that cannot be modified directly via settings. Users can provide extra instructions on top of the system prompt to customize agent behavior and how tasks should be performed, but user instructions cannot override security rules or core capabilities. Validates `instructions` to reject potential prompt injections or system overrides with HTTP 422.

- **Request Headers**: `Content-Type: application/json`
- **Request Body** (`SettingsUpdateRequest`, all fields optional):
  - `instructions` (*string, optional*): Global user instructions to steer task execution, behavior, tone, and formatting preferences. Subordinate to the system prompt and validated to ensure security guidelines and core capabilities cannot be overridden or bypassed.
  - `temperature` (*float, optional, 0.0 to 2.0*): Default sampling temperature.
  - `max_iterations` (*integer, optional, 1 to 50*): Default tool loop iteration ceiling.
  - `thinking` (*boolean, optional*): Default thinking / reasoning mode.
  - `reset_to_default` (*boolean, optional, default: false*): If `true`, clears custom user instructions and restores default runtime parameters.
- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8000/settings \
    -H "Content-Type: application/json" \
    -d '{
      "instructions": "Be concise, polite, and structure multi-part answers as bullet points.",
      "temperature": 0.5,
      "max_iterations": 12,
      "thinking": false
    }'
  ```
- **Response** (`200 OK`, `SettingsResponse`): Returns updated configuration.

---

### 3. Chat & AI Agent Endpoints

#### `POST /chat`
Executes synchronous conversational turns with multi-step tool execution, automatic memory persistence, and configurable thinking/reasoning mode. Global instructions and base system prompt are configured via `/settings` or `.env`.

- **Request Headers**: `Content-Type: application/json`
- **Request Body** (`ChatRequest`):
  - `message` (*string, required*): User query or prompt.
  - `temperature` (*float, optional, 0.0 to 2.0*): Sampling temperature override for this turn. When omitted, uses the configured default from `/settings`.
  - `max_iterations` (*integer, optional, 1 to 50*): Maximum tool execution loop iterations for this turn. When omitted, uses the configured default from `/settings`.
  - `thinking` (*boolean, optional*): Explicitly enable or disable thinking/reasoning mode for this turn. When omitted, uses the configured default from `/settings`.
  - `session_id` (*string, optional*): Existing session ID to resume conversation. If omitted, a new UUID is generated.
  - `stream_tools` (*boolean, optional*): Reserved for tool execution notices (default `false`).
- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{
      "message": "Calculate (245 * 18) / 7 + 33^2",
      "temperature": 0.5,
      "max_iterations": 8,
      "thinking": false,
      "session_id": "sess-user-01"
    }'
  ```
- **Response** (`200 OK`, `ChatResponse`):
  ```json
  {
    "response": "The result is 1719.0.",
    "model": "qwen3.5:9b",
    "session_id": "sess-user-01"
  }
  ```

#### `POST /chat/stream`
Streams generated assistant tokens over Server-Sent Events (SSE). Emits `X-Session-ID` in HTTP headers so clients can track the conversation thread. Model selection is strictly governed by `.env` (`MODEL`), and system instructions originate from `/settings`.

- **Request Headers**: `Content-Type: application/json`
- **Request Body** (`ChatRequest`):
  - `message` (*string, required*): User query or prompt.
  - `temperature` (*float, optional, 0.0 to 2.0*): Sampling temperature override for this turn.
  - `max_iterations` (*integer, optional, 1 to 50*): Maximum tool execution iterations.
  - `thinking` (*boolean, optional*): Set to `false` for immediate content streaming without waiting for reasoning tokens; set to `true` to enable deep reasoning.
  - `session_id` (*string, optional*): Session ID to resume or continue.
  - `stream_tools` (*boolean, optional*): If `true`, emits `[Calling tool '<tool_name>'...]\n` notices in the stream.
- **cURL Example**:
  ```bash
  curl -N -X POST http://localhost:8000/chat/stream \
    -H "Content-Type: application/json" \
    -d '{
      "message": "What is the current time in Asia/Kolkata?",
      "temperature": 0.3,
      "max_iterations": 6,
      "thinking": false,
      "stream_tools": true,
      "session_id": "sess-user-01"
    }'
  ```
- **Response** (`200 OK`, `text/event-stream`, Headers: `X-Session-ID: sess-user-01`):
  ```text
  [Calling tool 'get_current_time'...]
  The current time in Asia/Kolkata is 10:45 AM IST on November 21, 2026.
  ```

---

### 4. Conversations & Session Memory Endpoints

#### `GET /conversations`
Lists all active persisted conversation sessions, ordered by most recent message first.

- **Query Parameters**:
  - `limit` (*integer, optional, default: 50*): Maximum sessions to return.
  - `offset` (*integer, optional, default: 0*): Pagination offset.
- **cURL Example**:
  ```bash
  curl "http://localhost:8000/conversations?limit=10&offset=0"
  ```
- **Response** (`200 OK`, list of `ConversationSessionItem`):
  ```json
  [
    {
      "session_id": "sess-user-01",
      "summary": "Calculate arithmetic expression",
      "last_message_at": "2026-11-21T10:45:00Z",
      "created_at": "2026-11-21T10:40:00Z",
      "message_count": 4
    }
  ]
  ```

#### `GET /conversations/{session_id}`
Retrieves a specific conversation session along with its complete, turn-by-turn message history.

- **Path Parameters**: `session_id` (*string, required*)
- **cURL Example**:
  ```bash
  curl http://localhost:8000/conversations/sess-user-01
  ```
- **Response** (`200 OK`, `ConversationDetailResponse`):
  ```json
  {
    "session_id": "sess-user-01",
    "summary": "Calculate arithmetic expression",
    "last_message_at": "2026-11-21T10:45:00Z",
    "created_at": "2026-11-21T10:40:00Z",
    "message_count": 2,
    "messages": [
      {
        "role": "user",
        "content": "Calculate (245 * 18) / 7 + 33^2",
        "name": null,
        "tool_calls": null
      },
      {
        "role": "assistant",
        "content": "The result is 1719.0.",
        "name": null,
        "tool_calls": null
      }
    ]
  }
  ```

#### `DELETE /conversations/{session_id}`
Deletes a conversation session record and its associated history.

- **Path Parameters**: `session_id` (*string, required*)
- **cURL Example**:
  ```bash
  curl -X DELETE http://localhost:8000/conversations/sess-user-01
  ```
- **Response** (`204 No Content`)

---

### 5. Document Management & RAG Knowledge Base (`/documents`)

Persistent vector search uses ChromaDB under `data/chroma/`.

#### `POST /documents`
Ingests, chunks, and embeds documents into the production vector database. Supports both raw JSON text and multipart file uploads.

- **Option A: Raw Text via JSON (`Content-Type: application/json`)**:
  ```bash
  curl -X POST http://localhost:8000/documents \
    -H "Content-Type: application/json" \
    -d '{
      "content": "ShopNest notifies supervisory authorities within 72 hours of a personal data breach.",
      "doc_id": "privacy-policy-2026",
      "metadata": {"category": "compliance"}
    }'
  ```
- **Option B: File Upload (`multipart/form-data`)** (.pdf, .docx, .md, .txt):
  ```bash
  curl -X POST http://localhost:8000/documents \
    -F "file=@shopnest_privacy_policy.pdf" \
    -F "doc_id=privacy-policy-2026" \
    -F 'metadata={"category": "compliance"}'
  ```
- **Response** (`200 OK`, `DocumentIngestResponse`):
  ```json
  {
    "status": "success",
    "doc_id": "privacy-policy-2026",
    "chunks_count": 18,
    "message": "Successfully ingested 18 chunk(s) from 'shopnest_privacy_policy.pdf' into knowledge base."
  }
  ```

#### `GET /documents`
Lists all uploaded documents registered in the knowledge base.

- **cURL Example**:
  ```bash
  curl http://localhost:8000/documents
  ```
- **Response** (`200 OK`, list of `UploadedDocumentItem`):
  ```json
  [
    {
      "doc_id": "privacy-policy-2026",
      "filename": "shopnest_privacy_policy.pdf",
      "source": "shopnest_privacy_policy.pdf",
      "chunks_count": 18,
      "created_at": "2026-11-21T10:00:00Z"
    }
  ]
  ```

#### `GET /documents/chunks`
Fetches matching vector chunks for a search query using query parameters.

- **Query Parameters**:
  - `query` (*string, required*): Search query.
  - `k` (*integer, optional, default: 4*): Number of chunks to retrieve.
- **cURL Example**:
  ```bash
  curl "http://localhost:8000/documents/chunks?query=breach+notification+hours&k=2"
  ```
- **Response** (`200 OK`, `DocumentSearchResponse`):
  ```json
  {
    "query": "breach notification hours",
    "total_chunks": 2,
    "chunks": [
      {
        "chunk_index": 1,
        "content": "## Section: ShopNest Privacy Policy > 7. Security\n\n**Breach notification:** we notify affected users without undue delay, and within 72 hours to supervisory authorities.",
        "metadata": {
          "doc_id": "privacy-policy-2026",
          "filename": "shopnest_privacy_policy.pdf",
          "section": "ShopNest Privacy Policy > 7. Security"
        },
        "score": 0.912
      }
    ]
  }
  ```

#### `POST /documents/query` (or `POST /documents/chunks`)
Fetches matching vector chunks using a JSON payload.

- **Request Body** (`DocumentSearchRequest`):
  - `query` (*string, required*): Search query.
  - `k` (*integer, optional, default: 4*): Number of chunks.
- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8000/documents/query \
    -H "Content-Type: application/json" \
    -d '{"query": "data retention support chats", "k": 2}'
  ```
- **Response** (`200 OK`, `DocumentSearchResponse`):
  ```json
  {
    "query": "data retention support chats",
    "total_chunks": 1,
    "chunks": [
      {
        "chunk_index": 1,
        "content": "## Section: ShopNest Privacy Policy > 6. Data Retention\n\nCustomer support chats: retained 3 years. Call recordings: 12 months.",
        "metadata": {
          "doc_id": "privacy-policy-2026",
          "section": "ShopNest Privacy Policy > 6. Data Retention"
        },
        "score": 0.887
      }
    ]
  }
  ```

#### `DELETE /documents/{doc_id}`
Deletes a document entirely from ChromaDB (all embeddings) and removes its registry entry.

- **Path Parameters**: `doc_id` (*string, required*)
- **cURL Example**:
  ```bash
  curl -X DELETE http://localhost:8000/documents/privacy-policy-2026
  ```
- **Response** (`200 OK`, `DocumentDeleteResponse`):
  ```json
  {
    "status": "success",
    "doc_id": "privacy-policy-2026",
    "message": "Successfully deleted document 'privacy-policy-2026' and all its chunks from vector database."
  }
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

# Clean app.db, ChromaDB, and recreate all tables and migrations from scratch
make db-reset

# Clean app.db, ChromaDB, and re-apply existing migrations
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

## RAG Evaluation REST API (`/evals`) & Dashboard Backend

The agent exposes a dedicated `/evals` REST API router designed for dashboard panels to manage evaluation documents, trigger benchmarks, and retrieve metrics. All operations are strictly isolated from production databases (`data/eval/eval_chroma` and `data/eval/eval_app.db`).

### 1. Upload Documents to Evaluation Knowledge Base (`POST /evals/documents`)
Upload raw text or files (`.pdf`, `.docx`, `.txt`, `.md`) to the evaluation Chroma DB:
```bash
# Upload via JSON raw text
curl -X POST http://localhost:8000/evals/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "ChromaDB evaluation documentation...", "doc_id": "eval-doc-1"}'

# Upload via file (PDF, DOCX, TXT, MD)
curl -X POST http://localhost:8000/evals/documents \
  -F "file=@eval_manual.pdf" \
  -F "doc_id=manual-v1"
```

### 2. List Evaluation Documents (`GET /evals/documents`)
- **cURL Example**:
  ```bash
  curl http://localhost:8000/evals/documents
  ```
- **Response** (`200 OK`, list of `UploadedDocumentItem`):
  ```json
  [
    {
      "doc_id": "eval-doc-1",
      "filename": "eval_manual.pdf",
      "source": "eval_manual.pdf",
      "chunks_count": 5,
      "created_at": "2026-11-21T09:00:00Z"
    }
  ]
  ```

### 3. Fetch Evaluation Chunks (`GET /evals/chunks` or `POST /evals/query`)
Manually inspect matching chunks retrieved from the evaluation ChromaDB.

- **Option A: Via GET Query**:
  ```bash
  curl "http://localhost:8000/evals/chunks?query=evaluation&k=2"
  ```
- **Option B: Via POST JSON**:
  ```bash
  curl -X POST http://localhost:8000/evals/query \
    -H "Content-Type: application/json" \
    -d '{"query": "evaluation", "k": 2}'
  ```
- **Response** (`200 OK`, `DocumentSearchResponse`):
  ```json
  {
    "query": "evaluation",
    "total_chunks": 2,
    "chunks": [
      {
        "chunk_index": 1,
        "content": "## Section: Eval Guide\n\nEvaluation documents are isolated.",
        "metadata": {"doc_id": "eval-doc-1"},
        "score": 0.94
      }
    ]
  }
  ```

### 4. Delete Evaluation Document (`DELETE /evals/documents/{doc_id}`)
Cleanly removes a document and its chunks from the evaluation Chroma DB:
```bash
curl -X DELETE http://localhost:8000/evals/documents/eval-doc-1
```
- **Response** (`200 OK`, `DocumentDeleteResponse`):
  ```json
  {
    "status": "success",
    "doc_id": "eval-doc-1",
    "message": "Successfully deleted document 'eval-doc-1' and all its chunks from evaluation vector database."
  }
  ```

### 5. Run Evaluation Benchmark (`POST /evals/run`)
Triggers LLM-as-a-judge evaluation across **Correctness**, **Answer Relevance**, **Groundedness**, and **Retrieval Relevance**, automatically syncing to LangSmith when `LANGCHAIN_API_KEY` is configured:
```bash
curl -X POST http://localhost:8000/evals/run \
  -H "Content-Type: application/json" \
  -d '{"clean_on_start": false}'
```
*Response for Dashboard Panel:*
```json
{
  "status": "success",
  "timestamp": "2026-11-15T13:00:00Z",
  "dataset_name": "rag-agent-evaluation",
  "total_cases": 6,
  "metrics_summary": {
    "correctness": 0.967,
    "answer_relevance": 0.983,
    "groundedness": 0.950,
    "retrieval_relevance": 0.933,
    "overall_score": 0.958
  },
  "cases": [
    {
      "id": "tc-001",
      "question": "Where does ChromaDB persist vector embeddings by default?",
      "reference_answer": "Under the directory 'data/chroma' with collection name 'rag_documents'.",
      "agent_response": "ChromaDB persists vector embeddings in data/chroma...",
      "metrics": {
        "correctness": {"score": 1.0, "reasoning": "..."},
        "answer_relevance": {"score": 1.0, "reasoning": "..."},
        "groundedness": {"score": 1.0, "reasoning": "..."},
        "retrieval_relevance": {"score": 1.0, "reasoning": "..."}
      }
    }
  ]
}
```

### 6. Fetch Latest Evaluation Results (`GET /evals/results`)
Instantly retrieves the latest evaluation report for dashboard rendering without re-running tests:
```bash
curl http://localhost:8000/evals/results
```
- **Response** (`200 OK`, latest evaluation report JSON):
  ```json
  {
    "status": "success",
    "timestamp": "2026-11-21T10:45:56Z",
    "dataset_name": "rag-agent-evaluation",
    "total_cases": 1,
    "metrics_summary": {
      "overall_score": 0.96,
      "correctness": 0.95,
      "answer_relevance": 0.95,
      "groundedness": 0.95,
      "retrieval_relevance": 1.00
    }
  }
  ```

---

## Dataset Definition (`eval/test_cases.yaml`)
Benchmark test cases are configured in YAML (evaluation documents are uploaded via `POST /evals/documents`):
```yaml
test_cases:
  - id: "tc-001"
    question: "Where does ChromaDB persist vector embeddings by default?"
    reference_answer: "Under the directory 'data/chroma' with collection name 'rag_documents'."
    tags: ["rag", "chromadb"]
```

### 3. Strict Database Isolation
The evaluation harness uses **completely separate test databases**:
- SQLite: `data/eval/eval_app.db`
- ChromaDB: `data/eval/eval_chroma` (collection: `eval_rag_documents`)
- **Main databases (`data/app.db` and `data/chroma`) remain completely untouched.**

### 4. Running Evaluations
```bash
# Run local evaluation and benchmark
make eval

# Or with custom flags
.venv/bin/python -m eval.runner
```

When `LANGCHAIN_API_KEY` is present in your `.env`, the evaluation suite automatically uploads/syncs test cases to a **LangSmith dataset** (`rag-agent-evaluation`) and logs benchmark experiment results to LangSmith.

Detailed evaluation reports are generated in both JSON and Markdown:
- `eval/results/eval_report.json`
- `eval/results/eval_report.md`

---

## Running Tests
```bash
make test
make test-verbose
```
