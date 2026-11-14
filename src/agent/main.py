from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from ollama import Client

from src.agent.api.routes_chat import router as chat_router
from src.agent.api.routes_evals import router as evals_router
from src.agent.db.repositories import init
from src.agent.settings import SETTINGS, Settings


@asynccontextmanager
async def lifespan(application: FastAPI):
    await init()
    yield


def build_app(
    settings: Optional[Settings] = None,
    client: Optional[Client] = None,
) -> FastAPI:
    """
    Composition root: instantiates and wires the FastAPI application,
    attaching injected settings, client, and dependencies.
    """
    app_settings = settings or SETTINGS
    app_client = client or app_settings.OLLAMA_CLIENT

    app = FastAPI(lifespan=lifespan)
    app.state.settings = app_settings
    app.state.client = app_client

    app.include_router(chat_router)
    app.include_router(evals_router)

    for route in chat_router.routes:
        if route not in app.routes:
            app.routes.append(route)
    for route in evals_router.routes:
        if route not in app.routes:
            app.routes.append(route)

    return app


app = build_app()
client = SETTINGS.OLLAMA_CLIENT
