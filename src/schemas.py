from pydantic import BaseModel, Field
from .settings import SETTINGS

class Query(BaseModel):
    model: str = Field(default=SETTINGS.DEFAULT_MODEL)
    input: str

class Message(BaseModel):
    role: str
    content: str