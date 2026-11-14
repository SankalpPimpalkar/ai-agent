# AI Agent Architecture

## System Architecture

```mermaid
graph TD
    User([User / Client]) -->|HTTP POST| API[FastAPI Layer\nsrc/agent/main.py]
    API -->|Depends| Routes[Chat & Evals Routes\nsrc/agent/api/]
    Routes -->|Invoke| Agent[Agent Core\nsrc/agent/core/agent.py]
    Agent -->|Breakers| Breakers[LoopBreakers\nsrc/agent/core/breakers.py]
    Agent -->|Format| Prompt[PromptBuilder\nsrc/agent/conversation/]
    Agent -->|Manage| History[ConversationHistory\nsrc/agent/conversation/]
    Agent -->|Execute| Registry[ToolRegistry\nsrc/agent/tools/]
  
    Registry --> Calc[CalculatorTool]
    Registry --> Cache[CacheTool]
    Registry --> CLI[CliTool\nsrc/agent/tools/cli/]
    Registry --> Builder[ToolBuilderTool\nsrc/agent/tools/builder/]
    Registry --> Memory[MemoryTool & Service\nsrc/agent/tools/memory/]
    Registry --> Planner[PlannerTool]

    Agent -->|Persist State| DataDir[(data/\napp.db, artifacts, memory, agent-tools)]
    API -->|Load Prompts| DBRepo[PromptVersion Repositories\nsrc/agent/db/]
    DBRepo -->|SQLModel| SQLite[(data/app.db)]
```

## Directory Layout & Boundaries

- **`src/agent/`**: Single unified top-level application package.
  - **`core/`**: Central reasoning loop (`Agent`), circuit breakers (`LoopBreakers`), and state definitions (`AgentState`, `Trace`).
  - **`conversation/`**: Chat history, message trimming, and prompt assembling (`PromptBuilder`).
  - **`tools/`**: `BaseTool` abstractions, `ToolRegistry`, and focused tools decomposed into single-responsibility collaborators.
  - **`api/`**: Thin FastAPI route controllers, schemas, and dependency injection providers.
  - **`db/`**: SQLModel schemas, async session creation, and database repositories.
  - **`prompts/`**: Externalized plain text prompt definitions (`system.txt`, `tools.txt`, `memory.txt`, `plan.txt`).
- **`data/`**: Git-ignored runtime state including SQLite database, project artifacts, memory text files, and custom dynamic tools.
- **`evals/`**: Evaluation suite decomposed into `cases/`, `layers/`, runner, reporter, and storage.
- **`tests/`**: Unit tests mirroring `src/agent/`.
