from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalRunRequest(BaseModel):
    yaml_path: Optional[str] = Field(
        default=None,
        description="Optional custom path to YAML test dataset file. Defaults to eval/test_cases.yaml.",
    )
    clean_on_start: bool = Field(
        default=False,
        description="Whether to clear the evaluation Chroma and SQLite database before running.",
    )
    test_cases: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional dynamic test cases payload to evaluate directly from dashboard.",
    )


class EvalMetricsSummary(BaseModel):
    correctness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    answer_relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    groundedness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    retrieval_relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    total_cases: int = 0
    evaluated_cases: int = 0
    excluded_cases: int = 0
    score_display: Optional[str] = None
    metrics_display: Optional[Dict[str, str]] = None


class EvalCaseResult(BaseModel):
    id: str
    status: str = "success"
    error_message: Optional[str] = None
    question: str
    reference_answer: str
    agent_response: str
    retrieved_context: str
    tags: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any]


class EvalRunResponse(BaseModel):
    status: str = "success"
    timestamp: str
    dataset_name: str
    dataset_id: Optional[str] = None
    total_cases: int
    metrics_summary: EvalMetricsSummary
    cases: List[EvalCaseResult]
