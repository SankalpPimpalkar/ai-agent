from typing import List
from pydantic import BaseModel
from src.schemas import Trace, TraceArgs

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
