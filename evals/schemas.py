from pydantic import BaseModel
from typing import List

class TraceArgs(BaseModel):
    action: str | None = None
    model_config = {"extra": "allow"}

class Trace(BaseModel):
    name: str
    args: TraceArgs
    result: str

class Run(BaseModel):
    steps: int
    tokens: int
    answer: str
    trace: List[Trace]

class Case(BaseModel):
    name: str
    prompt: str
    max_steps: int
    tool_order: List[str]
    forbid_tools: List[str]