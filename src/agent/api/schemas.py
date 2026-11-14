from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Query(BaseModel):
    input: str
    model: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str


class ResponseData(BaseModel):
    answer: str
    steps: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int


class ApiResponse(BaseModel):
    data: ResponseData
