from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class EvalStatusResponse(BaseModel):
    status: str
    message: str


@router.get("/api/evals/status", response_model=EvalStatusResponse)
async def eval_status():
    return EvalStatusResponse(status="ready", message="Evals system configured.")
