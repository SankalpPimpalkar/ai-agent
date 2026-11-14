from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON, Text, Index, text, update
from typing import Dict, Any
from datetime import datetime, timezone
from sqlmodel.ext.asyncio.session import AsyncSession

class PromptVersion(SQLModel, table=True):
    __tablename__ = "prompt_versions"
    __table_args__ = (
        Index(
            "one_active_prompt",
            "is_active",
            unique=True,
            sqlite_where=text("is_active = 1"),
            postgresql_where=text("is_active"),
        ),
    )

    id: int | None = Field(primary_key=True, default=None)
    model: str
    is_active: bool = Field(default=False)
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))
    tool_instructions: str = Field(sa_column=Column(Text, nullable=False))
    memory_instructions: str = Field(sa_column=Column(Text, nullable=False))
    config_hash: str = Field(sa_column=Column(Text, nullable=False))
    settings: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<PromptVersion {self.id}>"


async def active(db: AsyncSession, version_id: int):
    await db.exec(
        update(PromptVersion).where(PromptVersion.is_active).values(is_active=False)
    )
    await db.exec(
        update(PromptVersion)
        .where(PromptVersion.id == version_id)
        .values(is_active=True)
    )
    await db.commit()
