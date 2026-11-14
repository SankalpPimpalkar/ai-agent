from typing import AsyncGenerator
from fastapi import Depends, Request
from langchain_core.vectorstores import VectorStore
from sqlmodel.ext.asyncio.session import AsyncSession
from app.agent.builder import AgentRunner, build_agent
from app.config import Settings, settings
from app.db import get_session
from app.rag.store import get_vector_store


def get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", settings)


async def get_agent(request: Request) -> AgentRunner:
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        return agent
    return await build_agent()


def get_vector_store_dep() -> VectorStore:
    return get_vector_store()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session():
        yield session
