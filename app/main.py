from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from app.agent.builder import AgentRunner, build_agent
from app.agent.memory import init_checkpointer
from app.api.routes.chat import router as chat_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.config import Settings, settings
from app.db import init_db
from app.schemas.chat import HealthResponse


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Initialize SQLModel database tables on startup
    await init_db()
    # Initialize checkpointer tables on SQLite database
    await init_checkpointer()
    yield


def build_app(
    app_settings: Optional[Settings] = None,
    agent_runner: Optional[AgentRunner] = None,
) -> FastAPI:
    """
    FastAPI application factory.
    Wires configuration, lifespan, dependencies, and route registrations.
    """
    active_settings = app_settings or settings

    application = FastAPI(
        title=active_settings.APP_NAME,
        lifespan=lifespan,
    )
    application.state.settings = active_settings

    if agent_runner is not None:
        application.state.agent = agent_runner

    # Health check route
    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health():
        return HealthResponse(
            status="ok",
            app_name=active_settings.APP_NAME,
            rag_enabled=active_settings.ENABLE_RAG,
            tools_enabled=active_settings.ENABLE_TOOLS,
            mcp_enabled=active_settings.ENABLE_MCP,
        )

    # Register API routers
    application.include_router(chat_router)
    application.include_router(conversations_router)
    application.include_router(documents_router)
    return application


app = build_app()
