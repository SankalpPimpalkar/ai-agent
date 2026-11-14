import os
from pathlib import Path
from typing import List, Optional
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.config import settings

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
