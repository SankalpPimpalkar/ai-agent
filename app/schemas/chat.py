from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or chat message")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Optional sampling temperature override (0.0 to 2.0)")
    max_iterations: Optional[int] = Field(default=None, ge=1, le=50, description="Optional maximum tool execution loop iterations override")
    session_id: Optional[str] = Field(default=None, description="Optional session or conversation ID")
    stream_tools: bool = Field(default=False, description="Whether to stream tool execution notices in streaming mode")
    thinking: Optional[bool] = Field(default=None, description="Whether to enable thinking / reasoning mode (controls thinking parameter for Ollama)")


class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: Optional[str] = None


class DocumentIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text content of the document to ingest")
    doc_id: Optional[str] = Field(default=None, description="Optional custom unique document ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata key-values for the document")


class DocumentIngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_count: int
    message: str


class DocumentDeleteResponse(BaseModel):
    status: str
    doc_id: str
    message: str


class UploadedDocumentItem(BaseModel):
    doc_id: str
    filename: str
    source: str
    chunks_count: int
    created_at: datetime


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    rag_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool


class ConversationSessionItem(BaseModel):
    session_id: str
    summary: str
    instructions: Optional[str] = None
    last_message_at: datetime
    created_at: datetime
    message_count: int


class MessageItem(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ConversationDetailResponse(ConversationSessionItem):
    messages: List[MessageItem] = []


class RetrievedChunkItem(BaseModel):
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query to retrieve matching chunks")
    k: int = Field(default=4, ge=1, le=50, description="Number of chunks to return")


class DocumentSearchResponse(BaseModel):
    query: str
    total_chunks: int
    chunks: List[RetrievedChunkItem]
