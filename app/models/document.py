from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


class UploadedDocument(SQLModel, table=True):
    __tablename__ = "uploaded_documents"

    doc_id: str = Field(primary_key=True, index=True, description="Unique document ID (UUID)")
    filename: str = Field(default="document.txt", description="Document filename or label")
    source: str = Field(default="manual", description="Document source or filepath")
    chunks_count: int = Field(default=0, description="Total chunks indexed in the vector store")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when document was ingested",
    )
