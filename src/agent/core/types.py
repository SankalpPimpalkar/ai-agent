from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ollama import ChatResponse, Message as OllamaMessage, Tool
from src.agent.core.breakers import LoopBreakers


class TraceArgs(BaseModel):
    action: Optional[str] = None
    model_config = {"extra": "allow"}


class Trace(BaseModel):
    name: str
    args: TraceArgs
    result: str
    step: Optional[int] = None


class ChatClientProtocol(Protocol):
    """
    Protocol abstracting the chat client (e.g. Ollama Client).
    """

    def chat(
        self,
        model: str,
        messages: List[OllamaMessage],
        tools: Optional[List[Tool]] = None,
        think: bool = False,
        options: Optional[dict] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        ...


@dataclass
class AgentState:
    steps: int = 0
    total_in: int = 0
    total_out: int = 0
    max_step_in: int = 0
    errors: int = 0
    seen: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    breakers: List[str] = field(default_factory=list)
    trace: List[Trace] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    answer: str = "Stopped after too many steps."
    _loop_breakers: LoopBreakers = field(default_factory=LoopBreakers)

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_in += prompt_tokens
        self.total_out += completion_tokens
        self.max_step_in = max(self.max_step_in, prompt_tokens)

    def check_breakers(
        self, name: str, args: dict, result: str, step: int
    ) -> str:
        updated_result = self._loop_breakers.check(name, args, result, step, self.logs)
        self.errors = self._loop_breakers.consecutive_errors
        self.seen = self._loop_breakers.seen_calls
        self.breakers = list(self._loop_breakers.triggered_breakers)
        return updated_result

    def to_dict(self) -> dict:
        total = self.total_in + self.total_out
        return {
            "answer": self.answer,
            "steps": self.steps,
            "trace": self.trace,
            "logs": self.logs,
            "breakers": self.breakers,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_tokens": total,
            "tokens": total,
            "max_step_prompt_tokens": self.max_step_in,
        }


def to_response(run: dict) -> JSONResponse:
    total_in = run.get("total_in", 0)
    total_out = run.get("total_out", 0)
    return JSONResponse(
        content={
            "data": {
                "answer": run.get("answer", ""),
                "steps": run.get("steps", 0),
                "total_tokens": total_in + total_out,
                "prompt_tokens": total_in,
                "completion_tokens": total_out,
            }
        }
    )
