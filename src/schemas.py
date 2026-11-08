from typing import Optional
from pydantic import BaseModel

class Query(BaseModel):
    input: str
    model: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class PromptVersionSettings(BaseModel):
    thinking: bool = False
    ctx_size: int = 8192

class PromptVersion(BaseModel):
    model: str
    is_active: bool
    system_prompt: str
    tool_instructions: str
    memory_instructions: str
    config_hash: str
    settings: PromptVersionSettings

class TraceArgs(BaseModel):
    action: Optional[str] = None
    model_config = {"extra": "allow"}

class Trace(BaseModel):
    name: str
    args: TraceArgs
    result: str
    step: Optional[int] = None
