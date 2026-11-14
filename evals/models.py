from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.agent.core.types import Trace, TraceArgs

class Run(BaseModel):
    steps: int
    tokens: int
    answer: str
    trace: List[Trace] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    total_in: Optional[int] = 0
    total_out: Optional[int] = 0

class Case(BaseModel):
    name: str
    prompt: str
    max_steps: int = 15
    forbid_tools: List[str] = Field(default_factory=list)
    expected_output: List[str] = Field(default_factory=list)
    tool_order: Optional[List[str]] = None

class CaseResult(BaseModel):
    case: str
    prompt: str
    passed: bool
    failures: Dict[str, List[str]] = Field(default_factory=dict)
    steps: int
    tokens: int
    total_in: int
    total_out: int
    answer: str
    trace: List[Trace] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)

class EvalRunRecord(BaseModel):
    run_id: str
    timestamp: str
    model: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    cases: List[CaseResult]
