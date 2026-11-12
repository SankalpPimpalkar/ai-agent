import hashlib
import json
from typing import Optional
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from src.db.main import engine
from src.db.models import PromptVersion, active
from src.settings import SETTINGS


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

