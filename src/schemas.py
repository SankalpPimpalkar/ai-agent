from pydantic import BaseModel

class Query(BaseModel):
    model: str
    input: str

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