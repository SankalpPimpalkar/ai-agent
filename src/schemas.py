from pydantic import BaseModel, Field
from .settings import DEFAULT_MODEL

class Query(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    input: str

class Message(BaseModel):
    role: str
    content: str