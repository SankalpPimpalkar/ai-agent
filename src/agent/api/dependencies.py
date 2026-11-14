from typing import Any
from fastapi import Depends, Request
from ollama import Client
from src.agent.settings import SETTINGS, Settings
from src.agent.tools.memory import MemoryService


def get_settings_dep(request: Request) -> Settings:
    return getattr(request.app.state, "settings", SETTINGS)


def get_client_dep(
    request: Request, settings: Settings = Depends(get_settings_dep)
) -> Client:
    return getattr(request.app.state, "client", settings.OLLAMA_CLIENT)


def get_agent_dep(
    client: Client = Depends(get_client_dep),
    settings: Settings = Depends(get_settings_dep),
) -> Any:
    from src.agent.core.agent import Agent
    return Agent(client=client, settings=settings)


def get_memory_service_dep(
    client: Client = Depends(get_client_dep),
    settings: Settings = Depends(get_settings_dep),
) -> MemoryService:
    return MemoryService(settings=settings, client=client)
