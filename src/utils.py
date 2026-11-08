from typing import List, Optional, Any
from .schemas import Query
from ollama import ChatResponse, Message
from .memory import messages, memory_pass
from src.settings import SETTINGS
from src.db.models import PromptVersion, active
from src.db.main import engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
import asyncio
import hashlib
import json

async def stream_response(client: Any, model: str, msgs: List[Any], query: Query):
    try:
        stream: ChatResponse = client.chat(
            model=model, messages=msgs, stream=True
        )
        full_response = ""

        for chunk in stream:
            out = chunk["message"]["content"]
            full_response += str(out)
            yield out

        messages.append(Message(role="assistant", content=full_response))
        asyncio.create_task(memory_pass(query.input, full_response, model=model))
    except Exception as e:
        yield f"\n[Error: {str(e)}]"

def _role(m):
    return m["role"] if isinstance(m, dict) else m.role

def trim_chat(chat_history: Optional[List[Message]] = None, max_messages: int = 20):
    history = messages if chat_history is None else chat_history
    if not history:
        return
    has_system = _role(history[0]) == "system"
    system_prompt = history[0] if has_system else None
    chat = history[1:] if has_system else history[:]

    if len(chat) > max_messages:
        chat = chat[-max_messages:]
        while chat and _role(chat[0]) != "user":
            chat.pop(0)

    history[:] = ([system_prompt] if system_prompt else []) + chat

def to_response(run: dict) -> dict:
    total_tokens = run.get("total_tokens", run.get("tokens", 0))
    return {
        "content": run.get("answer", ""),
        "usage": {
            "prompt_tokens": run.get("total_in", 0),
            "completion_tokens": run.get("total_out", 0),
            "total_tokens": total_tokens,
        },
    }

def build_config() -> dict:
    return {
        "model": SETTINGS.DEFAULT_MODEL,
        "system_prompt": SETTINGS.SYSTEM_PROMPT,
        "tool_instructions": SETTINGS.TOOL_INSTRUCTIONS,
        "memory_instructions": SETTINGS.MEMORY_INSTRUCTIONS,
        "settings": {
            "thinking": SETTINGS.THINKING,
            "ctx_size": SETTINGS.MODEL_CTX_SZ,
        },
    }

def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

async def init():
    cfg = build_config()
    h = config_hash(cfg)
    async with AsyncSession(engine) as db:
        result = await db.exec(select(PromptVersion).where(PromptVersion.config_hash == h))
        row = result.first()
        if not row:
            row = PromptVersion(**cfg, config_hash=h)
            db.add(row)
            await db.commit()
            await db.refresh(row)
        if not row.is_active:
            await active(db, row.id)

async def get_settings() -> Optional[PromptVersion]:
    async with AsyncSession(engine) as db:
        result = await db.exec(select(PromptVersion).where(PromptVersion.is_active == True))
        row = result.first()
        if not row:
            await init()
            result = await db.exec(select(PromptVersion).where(PromptVersion.is_active == True))
            row = result.first()
        return row
