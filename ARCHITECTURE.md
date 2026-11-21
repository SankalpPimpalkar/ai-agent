# System Architecture & Code Flow

This document details the architectural design, module responsibilities, security boundaries, and runtime execution flow for the AI Agent service.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Tooling Layer"]
        Client["HTTP Client / Browser / Frontend UI"]
        Bruno["Bruno API Collection (bruno/)"]
    end

    subgraph APILayer["FastAPI Application Layer (app/)"]
        Main["app/main.py\n(FastAPI Composition Root & Lifespan)"]
        ChatRoute["app/api/routes/chat.py\nPOST /chat\nPOST /chat/stream"]
        SettingsRoute["app/api/routes/settings.py\nGET /settings\nPOST /settings"]
        ConvRoute["app/api/routes/conversations.py\nGET /conversations\nGET /conversations/{id}\nDELETE /conversations/{id}"]
        DocRoute["app/api/routes/documents.py\nPOST /documents\nGET /documents\nGET/POST /documents/chunks\nDELETE /documents/{id}"]
        EvalRoute["app/api/routes/evals.py\nPOST/GET/DELETE /evals/documents\nGET/POST /evals/chunks\nPOST /evals/run\nGET /evals/results"]
        HealthRoute["GET /health"]
        Deps["app/api/deps.py\n(Dependency Injection)"]
    end

    subgraph SecurityLayer["Security & Prompt Guardrails (app/agent/security.py)"]
        SecValidator["validate_user_instructions()\n- Anti-Prompt Injection\n- Jailbreak & DAN Detection\n- System Prompt Override Blocks\n- Exfiltration & Destructive Blocks"]
    end

    subgraph AgentLayer["Agent & LLM Engine (app/agent/)"]
        Builder["app/agent/builder.py\n(AgentRunner, Message Builder & Tool Loop)"]
        LLM["app/agent/llm.py\n(ChatOllama Provider: Thinking & Params)"]
        Memory["app/agent/memory.py\n(Checkpointer & Session State)"]
        Middleware["app/agent/middleware.py\n(Tool Pruning & Summarization)"]
        SysPromptFile["system-prompt.txt\n(Immutable Abstract Baseline)"]
    end

    subgraph ToolLayer["Tool Ecosystem (app/tools/)"]
        Registry["app/tools/registry.py\n(Dynamic Aggregator)"]
        LocalTools["app/tools/local.py\n(Native @tool functions)"]
        MCPAdapter["app/tools/mcp.py\n(MultiServerMCPClient)"]
        RAGTool["app/tools/rag.py\n(Retriever as a Tool)"]
    end

    subgraph RAGLayer["RAG Subsystem (app/rag/)"]
        Ingest["app/rag/ingest.py\n(Section & Table Chunking)"]
        Store["app/rag/store.py\n(VectorStore: ChromaDB)"]
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
    Client -->|API Requests| Main
    Bruno -->|Collections| Main

    %% Route Distribution
    Main --> ChatRoute
    Main --> SettingsRoute
    Main --> ConvRoute
    Main --> DocRoute
    Main --> EvalRoute
    Main --> HealthRoute

    %% Route Security & Dependencies
    SettingsRoute --> SecValidator
    ChatRoute --> Deps
    DocRoute --> Deps
    SettingsRoute --> Deps
    Deps --> Builder
    Deps --> Store
    Deps --> DB

    %% Agent Construction & Hierarchy
    Builder --> SysPromptFile
    Builder --> LLM
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

### 2.1 Settings Configuration & Security Validation Flow

The `/settings` route provides dynamic runtime configuration for UI layers and clients while enforcing strict security boundaries over the abstract system prompt and user instructions:

```mermaid
sequenceDiagram
    autonumber
    actor Client as UI / Client / Admin
    participant Route as app/api/routes/settings.py
    participant Sec as app/agent/security.py
    participant Config as app/config.py (Settings)

    Client->>Route: POST /settings {"instructions": "...", "temperature": 0.2, "thinking": false}
    
    alt Instructions contain injection or system override
        Route->>Sec: validate_user_instructions(instructions)
        Sec-->>Route: Raise HTTPException(422) (Prompt Injection / Override detected)
        Route-->>Client: 422 Unprocessable Content {"detail": "Instruction rejected: Potential prompt injection..."}
    else Valid instructions
        Route->>Sec: validate_user_instructions(instructions)
        Sec-->>Route: Cleaned instructions
        Route->>Config: Update USER_INSTRUCTIONS, TEMPERATURE, MAX_ITERATIONS, ENABLE_REASONING
        Route-->>Client: 200 OK SettingsResponse (without exposing system_prompt)
    end
```

### 2.2 Chat Request Flow (with Instructions Hierarchy & Thinking Mode)

The sequence diagram below illustrates what happens from the moment a client sends a message to `/chat` or `/chat/stream` until the model responds (including prompt assembly, session checkpointer, middlewares, and automatic tool invocation):

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

    Client->>Router: POST /chat {"message": "...", "thinking": false, "session_id": "optional-uuid"}
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
  
    Router->>Runner: execute(message, session_id, temperature, max_iterations, thinking)
    Runner->>Runner: build_messages(): System prompt + Subordinate User Instructions
    Runner->>Memory: load_session_history(session_id)
    Memory-->>Runner: Prior conversation messages
    Runner->>Middlewares: prune_old_tool_outputs_middleware(messages)
    Middlewares-->>Runner: Messages with truncated old tool outputs
    Runner->>Middlewares: summarize_context_middleware(messages)
    Middlewares-->>Runner: Compressed messages (if context overflowed)

    Runner->>LLM: ainvoke(messages) with reasoning/thinking flag
  
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
| **`system-prompt.txt`**        | Abstract, immutable system prompt defining agent identity, tool capabilities, and security boundaries.   |
| **`bruno/`**                   | Declarative OpenCollection API suite for testing and inspecting endpoints across all modules in Bruno.  |
| **`mcp_servers.json`**         | Standard multi-platform MCP configuration. Holds server commands, arguments, and environment variables. |
| **`pyproject.toml`**           | Standard PEP 621 packaging and dependency metadata.                                                     |
| **`requirements.txt`**         | Pinned dependencies for virtual environment reproducibility.                                            |
| **`Makefile`**                 | CLI workflows: `make dev`, `make start`, `make test`, `make migrate`.                                   |
| **`alembic.ini`**              | Alembic CLI configuration pointing to `migrations/` and database URL.                                   |
| **`.env.sample`**              | Sample environment variables for developer configuration.                                               |
| **`ARCHITECTURE.md`**          | System architecture diagrams and technical specifications.                                              |
| **`README.md`**                | Overview, setup guide, and endpoint documentation.                                                      |

---

### 3.2 Application Source Code (`app/`)

#### Core Configuration & Application Lifespan

- **`app/main.py`**:
  Application factory (`build_app()`). Wires FastAPI, manages asynchronous startup/shutdown lifespan (`init_db`, `init_checkpointer`), mounts `/health`, `/chat`, `/chat/stream`, `/conversations`, `/documents`, `/evals`, and `/settings`.
- **`app/config.py`**:
  Typed application configuration powered by Pydantic's `BaseSettings`. Reads environment variables and flags (`ENABLE_RAG`, `ENABLE_TOOLS`, `ENABLE_MCP`, `MCP_CONFIG_PATH`, `MODEL`, `TEMPERATURE`, `MAX_ITERATIONS`, `USER_INSTRUCTIONS`).
- **`app/db.py`**:
  SQLModel database session provider. Sets up the async SQLAlchemy engine, handles SQLite/PostgreSQL connections, and exposes the `get_session()` dependency with `expire_on_commit=False`.

#### API Presentation Layer (`app/api/`)

- **`app/api/deps.py`**:
  FastAPI dependency injection providers: yields database sessions (`get_db_session`), assembles the agent runner (`get_agent`), retrieves application settings (`get_settings`), and retrieves the vector store (`get_vector_store_dep`).
- **`app/api/routes/chat.py`**:
  Centralized chat routes. Handles both synchronous JSON replies (`POST /chat`) and Server-Sent Events token streaming (`POST /chat/stream`), accepting per-request runtime overrides (`temperature`, `max_iterations`, `thinking`, `stream_tools`).
- **`app/api/routes/settings.py`**:
  Settings and instruction management (`GET /settings`, `POST /settings`). Allows dynamic runtime updates for `instructions`, `temperature`, `max_iterations`, and `thinking` mode. Enforces that the baseline system prompt is immutable over the API and never exposed in API responses.
- **`app/api/routes/conversations.py`**:
  Conversation history routes (`GET /conversations`, `GET /conversations/{session_id}`, `DELETE /conversations/{session_id}`) listing active sessions with their auto-updated topic summary and last message timestamp.
- **`app/api/routes/documents.py`**:
  RAG document management routes (`POST /documents`, `GET /documents`, `GET /documents/chunks`, `POST /documents/query`, `DELETE /documents/{doc_id}`). Supports both raw text payloads and file uploads in a single route, chunk debugging/manual retrieval inspection, lists indexed documents, and allows clean deletion from ChromaDB and the database registry.
- **`app/api/routes/evals.py`**:
  Dedicated evaluation REST API (`/evals`) for dashboard panels: document ingestion (`POST /evals/documents`), document listing (`GET /evals/documents`), chunk retrieval inspection (`GET /evals/chunks`, `POST /evals/query`), deletion (`DELETE /evals/documents/{doc_id}`), benchmark execution (`POST /evals/run`), and latest results retrieval (`GET /evals/results`) operating strictly on isolated evaluation databases.

#### Agent & Model Orchestration (`app/agent/`)

- **`app/agent/security.py`**:
  Instruction sanitization and prompt injection guardrail (`validate_user_instructions`). Evaluates custom user instructions against jailbreak attempts (`DAN mode`, `developer mode`), safety filter bypasses, credential/secret exfiltration (`cat .env`, prompt leaks), and system prompt override directives before applying them to the runtime.
- **`app/agent/llm.py`**:
  Instantiates the LangChain chat model (`ChatOllama`) configured with model name, base URL, temperature, and thinking/reasoning flags.
- **`app/agent/builder.py`**:
  `AgentRunner` class. Builds prompt sequences (`SystemMessage` + `HumanMessage`), binds tools, combines abstract system prompt with subordinate user instructions, and executes the multi-turn tool calling loop for both synchronous turns (`execute()`) and token streaming (`execute_stream()`) with optional tool status notices.
- **`app/agent/prompts.py`**:
  Default system prompt templates and chat message schemas.
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
  Document text extraction (`extract_text_from_file` supporting text/markdown, PDF via `pymupdf4llm` with table and Markdown layout extraction, `pdfplumber` structured table fallback, `pypdf`, and Word `.docx` with XML heading styles), section-aware text chunking (`chunk_text_by_sections` using subsection regex boundaries `split_subsections`, prepending heading breadcrumbs, and preserving Markdown table headers across chunk splits), vector store indexing with section metadata (`ingest_document`), and clean deletion (`delete_document_from_store`).
- **`app/rag/embeddings.py`**:
  Embeddings provider wrapping LangChain's `OllamaEmbeddings`.

#### Data Transfer Objects & Schemas (`app/schemas/` & `app/models/`)

- **`app/models/conversation.py`**:
  SQLModel table definition (`ConversationSession`) tracking `session_id`, auto-updated `summary`, `last_message_at`, `created_at`, and `message_count`.
- **`app/models/document.py`**:
  SQLModel table definition (`UploadedDocument`) tracking `doc_id`, `filename`, `source`, `chunks_count`, and `created_at`.
- **`app/schemas/chat.py`**:
  Pydantic models for chat requests (`ChatRequest`), responses (`ChatResponse`), document ingestion (`DocumentIngestRequest`, `DocumentIngestResponse`), document deletion (`DocumentDeleteResponse`, `UploadedDocumentItem`), conversation listings (`ConversationSessionItem`, `ConversationDetailResponse`), and health status.
- **`app/schemas/settings.py`**:
  Pydantic models for runtime settings updates (`SettingsUpdateRequest`) and configuration inspection (`SettingsResponse`).
- **`app/schemas/mcp.py`**:
  Strict schema models (`MCPServerConfig`, `MCPConfigFile`) validating `stdio` and `sse` MCP configurations, performing environment variable expansions, and enforcing platform standards.
- **`app/schemas/evals.py`**:
  Pydantic models for evaluation execution requests (`EvalRunRequest`), metrics summaries (`EvalMetricsSummary`), case outcomes (`EvalCaseResult`), and evaluation benchmark responses (`EvalRunResponse`).

#### Evaluation Subsystem (`eval/`)

- **`eval/test_cases.yaml`**:
  Declarative evaluation dataset containing seed knowledge base documents and question/reference answer test pairs with metadata and tags.
- **`eval/config.py`**:
  Evaluation configuration ensuring strict isolation with dedicated test databases (`data/eval/eval_app.db` and `data/eval/eval_chroma`).
- **`eval/dataset_loader.py`**:
  Loads YAML test cases and syncs dataset examples to LangSmith (`langsmith.Client`).
- **`eval/evaluators.py`**:
  LLM-as-a-judge evaluators for the 4 core RAG metrics:
  - **Correctness**: Compares agent output to reference answer.
  - **Answer Relevance**: Checks if the answer directly addresses the user question.
  - **Groundedness**: Verifies claims against retrieved context (hallucination detection).
  - **Retrieval Relevance**: Measures context precision of retrieved chunks.
- **`eval/runner.py`**:
  Orchestrates evaluation runs, outputs terminal summary tables, triggers LangSmith benchmarks, and exports reports to `eval/results/`.

---

## 4. Key Architectural Design Principles

1. **Standardized Protocol over Custom Protocols**:
   Uses the official **Model Context Protocol (MCP)** specification via `langchain-mcp-adapters` instead of building custom tool execution engines from scratch.
2. **Declarative Configuration**:
   MCP servers are added declaratively via `mcp_servers.json` without modifying Python code.
3. **Layered Instruction Hierarchy & Security Invariants**:
   The baseline system prompt is immutable over the API and loaded strictly from `system-prompt.txt`. User instructions can customize behavior and task execution, but are strictly subordinate to the system prompt and cannot override security rules or core capabilities.
4. **Active Prompt Injection Guardrails**:
   All user-provided instructions undergo strict regex and pattern verification (`app/agent/security.py`) to block prompt injections, jailbreaks, and data exfiltration attempts before execution.
5. **Pluggable Feature Flags**:
   Tools, RAG, and MCP can each be toggled on or off independently via boolean flags (`ENABLE_TOOLS`, `ENABLE_RAG`, `ENABLE_MCP`).
6. **Thin Web Layer**:
   FastAPI routes contain zero business logic; they validate DTO schemas, resolve dependencies, and delegate execution to `AgentRunner` or `ingest_document`.
7. **Robust Database Migrations**:
   Schema evolution is managed through Alembic async migrations (`alembic upgrade head`) referencing `SQLModel.metadata`.
