from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


class ConversationSession(SQLModel, table=True):
    __tablename__ = "conversation_sessions"

    session_id: str = Field(primary_key=True, index=True, description="Unique session ID")
    summary: str = Field(default="New Conversation", description="Auto-updated conversation topic summary")
    last_message_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
        description="Timestamp of the most recent message",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when conversation was initiated",
    )
    message_count: int = Field(default=0, description="Total count of messages in this session")
