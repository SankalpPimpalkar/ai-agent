from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or chat message")
    model: Optional[str] = Field(default=None, description="Optional LLM model override")
    system_prompt: Optional[str] = Field(default=None, description="Optional system prompt override")
    session_id: Optional[str] = Field(default=None, description="Optional session or conversation ID")
    stream_tools: bool = Field(default=False, description="Whether to stream tool execution notices in streaming mode")


class ChatResponse(BaseModel):
    response: str
    model: str
    session_id: Optional[str] = None


class DocumentIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text content of the document to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata key-values for the document")


class DocumentIngestResponse(BaseModel):
    status: str
    chunks_count: int
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    rag_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
