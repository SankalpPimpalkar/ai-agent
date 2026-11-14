from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.config import settings
from app.models.conversation import ConversationSession
from app.schemas.chat import MessageItem

_CUSTOM_CHECKPOINTER: Optional[BaseCheckpointSaver] = None


def set_custom_checkpointer(checkpointer: Optional[BaseCheckpointSaver]) -> None:
    """Sets an explicit checkpointer override (useful for tests, e.g. MemorySaver)."""
    global _CUSTOM_CHECKPOINTER
    _CUSTOM_CHECKPOINTER = checkpointer


def get_default_checkpointer(db_path: Optional[str] = None) -> Optional[BaseCheckpointSaver]:
    """Returns the custom checkpointer if configured, otherwise None (use SQLite context manager)."""
    return _CUSTOM_CHECKPOINTER


async def init_checkpointer(db_path: Optional[str] = None) -> None:
    """Initializes tables for SQLite checkpointer in the database."""
    if _CUSTOM_CHECKPOINTER is not None:
        if hasattr(_CUSTOM_CHECKPOINTER, "setup"):
            await _CUSTOM_CHECKPOINTER.setup()
        return

    path = db_path or settings.CHECKPOINTER_DB_PATH
    if path == ":memory:":
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()


async def load_session_history(
    session_id: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    db_path: Optional[str] = None,
) -> List[BaseMessage]:
    """
    Loads persisted conversation messages for a session ID using the checkpointer.
    """
    if not session_id:
        return []

    active_saver = checkpointer or _CUSTOM_CHECKPOINTER
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}

    if active_saver is not None:
        tuple_data = await active_saver.aget_tuple(config)
    else:
        path = db_path or settings.CHECKPOINTER_DB_PATH
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            tuple_data = await saver.aget_tuple(config)

    if (
        tuple_data
        and tuple_data.checkpoint
        and "channel_values" in tuple_data.checkpoint
        and "messages" in tuple_data.checkpoint["channel_values"]
    ):
        return list(tuple_data.checkpoint["channel_values"]["messages"])

    return []


async def save_session_history(
    session_id: str,
    messages: List[BaseMessage],
    checkpointer: Optional[BaseCheckpointSaver] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Persists updated conversation messages for a session ID using the checkpointer.
    """
    if not session_id or not messages:
        return

    active_saver = checkpointer or _CUSTOM_CHECKPOINTER
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}

    cp = empty_checkpoint()
    cp["channel_values"]["messages"] = messages
    cp["channel_versions"]["messages"] = 1
    new_versions = {"messages": 1}

    if active_saver is not None:
        await active_saver.aput(config, cp, {"source": "agent_session"}, new_versions)
    else:
        path = db_path or settings.CHECKPOINTER_DB_PATH
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            await saver.aput(config, cp, {"source": "agent_session"}, new_versions)


async def generate_conversation_topic(
    messages: List[BaseMessage],
    latest_user_message: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Generates a concise topic summary for the latest conversation topic.
    Extracts the user's latest query subject cleanly.
    """
    user_prompt = latest_user_message
    if not user_prompt:
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and m.content:
                user_prompt = m.content
                break

    if not user_prompt:
        return "New Conversation"

    clean = user_prompt.strip().replace("\n", " ")
    if len(clean) > 60:
        return clean[:57].rstrip() + "..."
    return clean or "Conversation"


async def record_session_metadata(
    session_id: str,
    messages: List[BaseMessage],
    latest_user_message: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    db_engine: Optional[AsyncEngine] = None,
) -> str:
    """
    Updates or inserts the ConversationSession record in SQLite:
    - Generates/updates the topic summary to reflect the latest topic.
    - Updates last_message_at timestamp to now.
    - Updates message_count.
    """
    if not session_id:
        return ""

    summary = await generate_conversation_topic(messages, latest_user_message, llm)

    from app.db import engine as default_engine
    eng = db_engine or default_engine

    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        statement = select(ConversationSession).where(ConversationSession.session_id == session_id)
        result = await session.exec(statement)
        record = result.first()
        if record is None:
            record = ConversationSession(
                session_id=session_id,
                summary=summary,
                last_message_at=now,
                created_at=now,
                message_count=len(messages),
            )
            session.add(record)
        else:
            record.summary = summary
            record.last_message_at = now
            record.message_count = len(messages)
        await session.commit()

    return summary


async def list_conversation_sessions(
    limit: int = 50,
    offset: int = 0,
    db_engine: Optional[AsyncEngine] = None,
) -> List[ConversationSession]:
    """Lists conversation sessions ordered by most recent message first."""
    from app.db import engine as default_engine
    eng = db_engine or default_engine

    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = (
            select(ConversationSession)
            .order_by(ConversationSession.last_message_at.desc())
            .offset(offset)
            .limit(limit)
        )
        results = await session.exec(statement)
        return list(results.all())


async def get_conversation_session(
    session_id: str,
    db_engine: Optional[AsyncEngine] = None,
) -> Optional[ConversationSession]:
    """Gets a specific conversation session record by session_id."""
    from app.db import engine as default_engine
    eng = db_engine or default_engine

    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(ConversationSession).where(ConversationSession.session_id == session_id)
        result = await session.exec(statement)
        return result.first()


async def delete_conversation_session(
    session_id: str,
    db_engine: Optional[AsyncEngine] = None,
) -> bool:
    """Deletes a conversation session record by session_id."""
    from app.db import engine as default_engine
    eng = db_engine or default_engine

    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(ConversationSession).where(ConversationSession.session_id == session_id)
        result = await session.exec(statement)
        record = result.first()
        if record:
            await session.delete(record)
            await session.commit()
            return True
        return False


def format_message_for_api(msg: BaseMessage) -> MessageItem:
    """Converts a LangChain BaseMessage to a serializable MessageItem schema."""
    role = (
        "user"
        if isinstance(msg, HumanMessage)
        else "assistant"
        if isinstance(msg, AIMessage)
        else "tool"
        if isinstance(msg, ToolMessage)
        else "system"
    )
    return MessageItem(
        role=role,
        content=str(getattr(msg, "content", "")),
        name=getattr(msg, "name", None),
        tool_calls=getattr(msg, "tool_calls", None),
    )
