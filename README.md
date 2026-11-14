# AI Agent (FastAPI + LangChain + SQLModel + Alembic)

A streamlined, production-ready AI chat service scaffolded with **FastAPI**, **LangChain** (via `ChatOllama`), **SQLModel**, and **Alembic** migrations.

## Features

- **LangChain Integration**: Leverages `ChatOllama` and standard LangChain messages (`HumanMessage`, `AIMessage`, `SystemMessage`) for clean model interactions.
- **Centralized Chat Endpoint**: Single endpoint at `POST /api/chat` handling conversation flows, optional session persistence, and Server-Sent Events (SSE) streaming (`stream: true`).
- **SQLModel Persistence**: Async SQLite / PostgreSQL session handling storing chat messages and prompt versions.
- **Alembic Migrations**: Fully configured async migration pipeline (`alembic upgrade head`) wired to SQLModel metadata.
- **Typed Configuration**: Settings managed via `pydantic-settings` with `.env` override support.

---

## Project Structure

```
├── Makefile                 # Development, migration, and testing workflows
├── alembic.ini              # Alembic migration configuration
├── pyproject.toml / requirements.txt
├── .env.sample              # Environment variable template
├── .gitignore
├── README.md
│
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── main.py          # FastAPI application factory and lifespan
│       ├── settings.py      # Typed settings and environment configuration
│       ├── core/
│       │   ├── __init__.py
│       │   └── service.py   # LangChain ChatService (ainvoke / astream)
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes_chat.py # Centralized /api/chat endpoint
│       │   ├── schemas.py   # Request and response models
│       │   └── dependencies.py # FastAPI dependency injection
│       └── db/
│           ├── __init__.py
│           ├── models.py    # SQLModel models (PromptVersion, ChatMessage)
│           ├── session.py   # Async database engine and session generator
│           └── repositories.py # Async database queries and persistence
│
├── migrations/              # Alembic migration scripts
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│
├── tests/                   # Automated unit test suite
│   ├── test_api.py          # API route tests (/health, /api/chat)
│   ├── test_db.py           # SQLModel database operations tests
│   └── test_service.py      # LangChain ChatService tests
│
└── data/                    # Runtime storage (gitignored)
    └── app.db               # SQLite database
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- Ollama running locally (`ollama run gemma4:12b` or your chosen model)

### 2. Setup
```bash
# Clone the repository and enter directory
cd ai-agent

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
make install

# Configure environment
cp .env.sample .env
```

### 3. Database Migrations
Apply database migrations using Alembic:
```bash
make migrate
```

### 4. Run the Application
```bash
# Start development server with auto-reload
make dev

# Or run on production host:port
make start
```
The API documentation is accessible at: `http://localhost:8000/docs`

---

## API Endpoints

### 1. Health Check
`GET /health`
```json
{
  "status": "ok",
  "app_name": "ai-agent"
}
```

### 2. Centralized Chat Endpoint
`POST /api/chat`

**Request Body:**
```json
{
  "message": "Hello, how does LangChain manage chat prompts?",
  "session_id": "user-session-123",
  "stream": false
}
```

**Response Body:**
```json
{
  "response": "LangChain manages chat prompts by formatting a sequence of messages (SystemMessage, HumanMessage, AIMessage)...",
  "session_id": "user-session-123",
  "model": "gemma4:12b"
}
```

**Streaming Request (SSE):**
Set `"stream": true` in the JSON payload to receive Server-Sent Events (`text/event-stream`).

---

## Running Tests
Run the test suite:
```bash
make test
```
Or for verbose test output:
```bash
make test-verbose
```
