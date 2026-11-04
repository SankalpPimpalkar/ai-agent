from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON, Text, Index, text, update
from typing import Dict, Any
from datetime import datetime, timezone
from sqlmodel import Session

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
    system_prompt: str = Field(sa_column=Column(Text, nullable=False))
    model: str
    settings: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    is_active: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<PromptVersion {self.id}>"


def active(db: Session, version_id: int):
    db.exec(
        update(PromptVersion).where(PromptVersion.is_active).values(is_active=False)
    )
    db.exec(
        update(PromptVersion)
        .where(PromptVersion.id == version_id)
        .values(is_active=True)
    )
    db.commit()
