# System Architecture & Code Flow

This document details the architectural design, module responsibilities, and runtime execution flow for the AI Agent service.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["Client Layer"]
        Client["HTTP Client / Browser / Frontend"]
    end

    subgraph APILayer["FastAPI Application Layer (app/)"]
        Main["app/main.py\n(FastAPI Composition Root & Lifespan)"]
        ChatRoute["app/api/routes/chat.py\nPOST /chat\nPOST /chat/stream"]
        ConvRoute["app/api/routes/conversations.py\nGET /conversations"]
        DocRoute["app/api/routes/documents.py\nPOST /documents"]
        HealthRoute["GET /health"]
        Deps["app/api/deps.py\n(Dependency Injection)"]
    end

    subgraph AgentLayer["Agent & LLM Engine (app/agent/)"]
        Builder["app/agent/builder.py\n(AgentRunner & Tool Loop)"]
        LLM["app/agent/llm.py\n(ChatOllama Provider)"]
        Prompts["app/agent/prompts.py\n(System & Chat Templates)"]
        Memory["app/agent/memory.py\n(Checkpointer & Session State)"]
        Middleware["app/agent/middleware.py\n(Tool Pruning & Summarization)"]
    end

    subgraph ToolLayer["Tool Ecosystem (app/tools/)"]
        Registry["app/tools/registry.py\n(Dynamic Aggregator)"]
        LocalTools["app/tools/local.py\n(Native @tool functions)"]
        MCPAdapter["app/tools/mcp.py\n(MultiServerMCPClient)"]
        RAGTool["app/tools/rag.py\n(Retriever as a Tool)"]
    end

    subgraph RAGLayer["RAG Subsystem (app/rag/)"]
        Ingest["app/rag/ingest.py\n(Chunking & Indexing)"]
        Store["app/rag/store.py\n(VectorStore: InMemory/Chroma)"]
        Embeddings["app/rag/embeddings.py\n(OllamaEmbeddings)"]
    end

    subgraph ExternalServices["External Services & Processes"]
        OllamaLLM["Ollama Service\n(localhost:11434)"]
        MCPGit["Git MCP Server\n(mcp-server-git via stdio)"]
        MCPCustom["Custom MCP Servers\n(stdio / sse)"]
    end

    subgraph DataLayer["Persistence Layer (app/db.py & Alembic)"]
        DB["app/db.py\n(SQLModel Async Engine & Sessions)"]
        SQLite[(data/app.db)]
        Alembic["migrations/\n(Schema Versioning)"]
    end

    %% Client Connections
    Client -->|POST /chat| Main
    Client -->|POST /chat/stream| Main
    Client -->|POST /documents| Main
    Client -->|GET /health| Main

    %% Route Distribution
    Main --> ChatRoute
    Main --> DocRoute
    Main --> HealthRoute

    %% Dependencies
    ChatRoute --> Deps
    DocRoute --> Deps
    Deps --> Builder
    Deps --> Store
    Deps --> DB

    %% Agent Construction
    Builder --> LLM
    Builder --> Prompts
    Builder --> Registry
    LLM --> OllamaLLM

    %% Tool Routing
    Registry --> LocalTools
    Registry --> RAGTool
    Registry --> MCPAdapter

    %% MCP Execution
    MCPAdapter -->|stdio| MCPGit
    MCPAdapter -->|stdio / sse| MCPCustom

    %% RAG Pipeline
    DocRoute --> Ingest
    Ingest --> Store
    Store --> Embeddings
    Embeddings --> OllamaLLM
    RAGTool --> Store

    %% Database
    DB --> SQLite
    Alembic --> SQLite
```

---

## 2. End-to-End Code Flow

### 2.1 Chat Request Flow (with Memory & Tool Calling)

The sequence diagram below illustrates what happens from the moment a client sends a message to `/chat` until the model responds (including session checkpointer, middlewares, and automatic tool invocation):

```mermaid
sequenceDiagram
    autonumber
    actor Client as Client / User
    participant Router as app/api/routes/chat.py
    participant Deps as app/api/deps.py
    participant Runner as app/agent/builder.py (AgentRunner)
    participant Memory as app/agent/memory.py (Checkpointer)
    participant Middlewares as app/agent/middleware.py
    participant Registry as app/tools/registry.py
    participant MCP as app/tools/mcp.py
    participant LLM as app/agent/llm.py (ChatOllama)
    participant ExtMCP as Git / External MCP Server

    Client->>Router: POST /chat {"message": "...", "session_id": "optional-uuid"}
    Router->>Router: Resolve session_id (reuse or generate UUID)
    Router->>Deps: get_agent()
    Deps->>Runner: build_agent()
    Runner->>Registry: get_all_tools()
  
    opt ENABLE_MCP is True
        Registry->>MCP: get_mcp_tools()
        MCP->>ExtMCP: Connect via stdio (MultiServerMCPClient)
        ExtMCP-->>MCP: [git_status, git_diff, ...]
        MCP-->>Registry: Return LangChain BaseTools
    end
  
    Registry-->>Runner: Return all active tools
    Runner->>LLM: bind_tools(tools)
  
    Router->>Runner: execute(message, session_id)
    Runner->>Memory: load_session_history(session_id)
    Memory-->>Runner: Prior conversation messages
    Runner->>Middlewares: prune_old_tool_outputs_middleware(messages)
    Middlewares-->>Runner: Messages with truncated old tool outputs
    Runner->>Middlewares: summarize_context_middleware(messages)
    Middlewares-->>Runner: Compressed messages (if context overflowed)

    Runner->>LLM: ainvoke(messages)
  
    alt LLM returns Tool Calls
        LLM-->>Runner: AIMessage(tool_calls=[git_status()])
        Note over Runner,ExtMCP: Tool Execution Loop (up to max_iterations)
        Runner->>ExtMCP: ainvoke git_status()
        ExtMCP-->>Runner: Output: "On branch main, working tree clean"
        Runner->>Runner: Append ToolMessage(content=..., tool_call_id=...)
        Runner->>LLM: ainvoke(messages + ToolMessage)
    end

    LLM-->>Runner: AIMessage("Final answer response")
    Runner->>Memory: save_session_history(session_id, updated_messages)
    Runner-->>Router: "Final answer response"
    Router-->>Client: 200 OK {"response": "...", "model": "...", "session_id": "uuid"}
```

---

### 2.2 RAG Ingestion, Retrieval & Deletion Flow

When document indexing and knowledge base retrieval are enabled (`ENABLE_RAG=True`), `POST /documents` supports both **raw JSON text** and **multipart file uploads** (`.txt`, `.md`, `.pdf`, `.docx`), assigning each upload a unique `doc_id`. Ingested chunks are tagged and persisted in **ChromaDB** on disk (`data/chroma`). In addition, `DELETE /documents/{doc_id}` allows clean, total removal of any document and its chunk vectors.

```mermaid
sequenceDiagram
    autonumber
    actor Admin as Admin / Ingestion Client
    participant DocRoute as app/api/routes/documents.py
    participant Ingest as app/rag/ingest.py
    participant DB as SQLite (UploadedDocument)
    participant Embeddings as app/rag/embeddings.py (OllamaEmbeddings)
    participant Chroma as ChromaDB (data/chroma)

    alt Document Ingestion (POST /documents)
        Admin->>DocRoute: POST /documents (JSON text or multipart file)
        DocRoute->>DocRoute: Assign unique doc_id (UUID or provided)
        DocRoute->>Ingest: extract_text_from_file(file) or raw text
        DocRoute->>Ingest: chunk_text(content) -> chunks tagged with doc_id
        Ingest->>Embeddings: aembed_documents(chunks)
        Embeddings-->>Ingest: Embeddings vectors
        Ingest->>Chroma: aadd_documents(chunks, ids=["doc_id_0", ...])
        Ingest->>DB: record_uploaded_document(doc_id, filename, chunks_count)
        DocRoute-->>Admin: 200 OK {"status": "success", "doc_id": "...", "chunks_count": N}
    else Document Deletion (DELETE /documents/{doc_id})
        Admin->>DocRoute: DELETE /documents/{doc_id}
        DocRoute->>Chroma: adelete(where={"doc_id": doc_id}) / adelete(ids)
        DocRoute->>DB: delete_uploaded_document_record(doc_id)
        DocRoute-->>Admin: 200 OK {"status": "success", "doc_id": "...", "message": "..."}
    end
```

---

## 3. Directory & File Breakdown

### 3.1 Root Files

| File                           | Purpose                                                                                                 |
| :----------------------------- | :------------------------------------------------------------------------------------------------------ |
| **`mcp_servers.json`** | Standard multi-platform MCP configuration. Holds server commands, arguments, and environment variables. |
| **`pyproject.toml`**   | Standard PEP 621 packaging and dependency metadata.                                                     |
| **`requirements.txt`** | Pinned dependencies for virtual environment reproducibility.                                            |
| **`Makefile`**         | CLI workflows: `make dev`, `make start`, `make test`, `make migrate`.                           |
| **`alembic.ini`**      | Alembic CLI configuration pointing to `migrations/` and database URL.                                 |
| **`.env.sample`**      | Sample environment variables for developer configuration.                                               |
| **`ARCHITECTURE.md`**  | System architecture diagrams and technical specifications.                                              |
| **`README.md`**        | Overview, setup guide, and endpoint documentation.                                                      |

---

### 3.2 Application Source Code (`app/`)

#### Core Configuration & Application Lifespan

- **`app/main.py`**:
  Application factory (`build_app()`). Wires FastAPI, manages asynchronous startup/shutdown lifespan (`init_db`), mounts `/health`, `/chat`, `/chat/stream`, and `/documents`.
- **`app/config.py`**:
  Typed application configuration powered by Pydantic's `BaseSettings`. Reads environment variables and flags (`ENABLE_RAG`, `ENABLE_TOOLS`, `ENABLE_MCP`, `MCP_CONFIG_PATH`).
- **`app/db.py`**:
  SQLModel database session provider. Sets up the async SQLAlchemy engine, handles SQLite/PostgreSQL connections, and exposes the `get_session()` dependency with `expire_on_commit=False`.

#### API Presentation Layer (`app/api/`)

- **`app/api/deps.py`**:
  FastAPI dependency injection providers: yields database sessions (`get_db_session`), assembles the agent runner (`get_agent`), and retrieves the vector store (`get_vector_store_dep`).
- **`app/api/routes/chat.py`**:
  Single centralized chat route. Handles both synchronous JSON replies (`POST /chat`) and Server-Sent Events token streaming (`POST /chat/stream`).
- **`app/api/routes/conversations.py`**:
  Conversation history routes (`GET /conversations`, `GET /conversations/{session_id}`, `DELETE /conversations/{session_id}`) listing active sessions with their auto-updated topic summary and last message timestamp.
- **`app/api/routes/documents.py`**:
  RAG document management routes (`POST /documents`, `GET /documents`, `DELETE /documents/{doc_id}`). Supports both raw text payloads (`application/json`) and file uploads (`multipart/form-data`) in a single route, assigns/accepts a unique `doc_id`, lists indexed documents, and allows clean deletion from ChromaDB and the database registry.

#### Agent & Model Orchestration (`app/agent/`)

- **`app/agent/llm.py`**:
  Instantiates the LangChain chat model (`ChatOllama`) configured with model name, base URL, and temperature.
- **`app/agent/builder.py`**:
  `AgentRunner` class. Builds prompt sequences (`SystemMessage` + `HumanMessage`), binds tools, and executes the multi-turn tool calling loop for both synchronous turns (`execute()`) and token streaming (`execute_stream()`) with optional tool status notices.
- **`app/agent/prompts.py`**:
  Default system prompts and `ChatPromptTemplate` structures.
- **`app/agent/memory.py`**:
  Persistent conversation memory backed by LangGraph checkpointers (`AsyncSqliteSaver`) storing multi-turn history indexed by `session_id` in SQLite.
- **`app/agent/middleware.py`**:
  Custom agent middlewares:
  - `prune_old_tool_outputs_middleware`: Replaces previous turns' tool outputs with concise placeholders while keeping schema-valid tool call identifiers.
  - `summarize_context_middleware`: Compresses earlier conversation history when context size overflows, preserving key facts and recent messages.

#### Tooling Subsystem (`app/tools/`)

- **`app/tools/registry.py`**:
  Central aggregator (`get_all_tools()`). Checks feature flags and dynamically combines native local tools, RAG retriever tools, and external MCP tools into a unified list.
- **`app/tools/local.py`**:
  Native Python functions decorated with `@tool` (e.g. `get_current_time`, `calculate_expression`).
- **`app/tools/mcp.py`**:
  Model Context Protocol client adapter. Parses `mcp_servers.json`, connects to external servers via `MultiServerMCPClient`, and returns remote tools as LangChain tools.
- **`app/tools/rag.py`**:
  `search_knowledge_base` `@tool` wrapping the vector store retriever, allowing the agent to perform semantic searches over indexed documents.

#### Retrieval-Augmented Generation (`app/rag/`)

- **`app/rag/store.py`**:
  Persistent vector store provider backed by **ChromaDB** (`langchain-chroma`). Automatically persists embeddings to disk under `data/chroma/`, with dependency injection support for test overrides.
- **`app/rag/ingest.py`**:
  Document text extraction (`extract_text_from_file` supporting text/markdown, PDF via `pypdf`, and Word `.docx` via standard XML parsing), text chunking (`chunk_text`), vector store indexing (`ingest_document`), and clean deletion (`delete_document_from_store`).
- **`app/rag/embeddings.py`**:
  Embeddings provider wrapping LangChain's `OllamaEmbeddings`.

#### Data Transfer Objects & Schemas (`app/schemas/` & `app/models/`)

- **`app/models/conversation.py`**:
  SQLModel table definition (`ConversationSession`) tracking `session_id`, auto-updated `summary`, `last_message_at`, `created_at`, and `message_count`.
- **`app/models/document.py`**:
  SQLModel table definition (`UploadedDocument`) tracking `doc_id`, `filename`, `source`, `chunks_count`, and `created_at`.
- **`app/schemas/chat.py`**:
  Pydantic models for chat requests (`ChatRequest`), responses (`ChatResponse`), document ingestion (`DocumentIngestRequest`, `DocumentIngestResponse`), document deletion (`DocumentDeleteResponse`, `UploadedDocumentItem`), conversation listings (`ConversationSessionItem`, `ConversationDetailResponse`), and health status.
- **`app/schemas/mcp.py`**:
  Strict schema models (`MCPServerConfig`, `MCPConfigFile`) validating `stdio` and `sse` MCP configurations, performing environment variable expansions, and enforcing platform standards.

---

## 4. Key Architectural Design Principles

1. **Standardized Protocol over Custom Protocols**:
   Uses the official **Model Context Protocol (MCP)** specification via `langchain-mcp-adapters` instead of building custom tool execution engines from scratch.
2. **Declarative Configuration**:
   MCP servers are added declaratively via `mcp_servers.json` without modifying Python code.
3. **Pluggable Feature Flags**:
   Tools, RAG, and MCP can each be toggled on or off independently via boolean flags (`ENABLE_TOOLS`, `ENABLE_RAG`, `ENABLE_MCP`).
4. **Thin Web Layer**:
   FastAPI routes contain zero business logic; they validate DTO schemas, resolve dependencies, and delegate execution to `AgentRunner` or `ingest_document`.
5. **Robust Database Migrations**:
   Schema evolution is managed through Alembic async migrations (`alembic upgrade head`) referencing `SQLModel.metadata`.
