from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.agent.builder import AgentRunner
from app.api.deps import get_agent, get_settings
from app.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


async def _chat_stream_generator(
    agent: AgentRunner,
    request: ChatRequest,
    model: str,
    system_prompt: str,
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in agent.execute_stream(
            message=request.message,
            system_prompt=system_prompt,
            model=model,
            stream_tools=request.stream_tools,
        ):
            yield chunk
    except Exception as e:
        yield f"\n[Error: {str(e)}]"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: AgentRunner = Depends(get_agent),
    settings: Settings = Depends(get_settings),
):
    """
    Synchronous chat endpoint returning the complete assistant response.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    model = request.model or settings.MODEL
    system_prompt = request.system_prompt or settings.SYSTEM_PROMPT

    response_text = await agent.execute(
        message=request.message,
        system_prompt=system_prompt,
        model=model,
    )

    return ChatResponse(
        response=response_text,
        model=model,
        session_id=request.session_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    agent: AgentRunner = Depends(get_agent),
    settings: Settings = Depends(get_settings),
):
    """
    Streaming chat endpoint returning tokens via Server-Sent Events (SSE).
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    model = request.model or settings.MODEL
    system_prompt = request.system_prompt or settings.SYSTEM_PROMPT

    return StreamingResponse(
        _chat_stream_generator(
            agent=agent,
            request=request,
            model=model,
            system_prompt=system_prompt,
        ),
        media_type="text/event-stream",
    )
