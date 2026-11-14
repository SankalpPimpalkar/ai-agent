import asyncio
from typing import Any, List
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ollama import Client, Message
from src.agent.api.dependencies import (
    get_agent_dep,
    get_client_dep,
    get_memory_service_dep,
    get_settings_dep,
)
from src.agent.api.schemas import Query
from src.agent.conversation.history import ConversationHistory, trim_chat
from src.agent.conversation.prompt_builder import PromptBuilder
from src.agent.core.types import to_response
from src.agent.db.repositories import get_settings
from src.agent.settings import Settings
from src.agent.tools.memory import MemoryService

router = APIRouter()
default_chat_history: List[Message] = []


async def stream_response(client: Any, model: str, msgs: List[Any], query: Query, memory_service: MemoryService):
    try:
        stream = client.chat(model=model, messages=msgs, stream=True)
        full_response = ""

        for chunk in stream:
            out = chunk["message"]["content"]
            full_response += str(out)
            yield out

        default_chat_history.append(Message(role="assistant", content=full_response))
        asyncio.create_task(memory_service.memory_pass(query.input, full_response, model=model))
    except Exception as e:
        yield f"\n[Error: {str(e)}]"


@router.post("/api/stream")
async def stream_chat(
    query: Query,
    client: Client = Depends(get_client_dep),
    settings: Settings = Depends(get_settings_dep),
    memory_service: MemoryService = Depends(get_memory_service_dep),
):
    db_settings = await get_settings()
    model = query.model or (
        db_settings.model if db_settings else settings.DEFAULT_MODEL
    )
    trim_chat(default_chat_history)
    default_chat_history.append(Message(role="user", content=query.input))

    builder = PromptBuilder(settings)
    formatted_msgs = builder.build_messages(
        chat_history=default_chat_history, tool_use=False
    )

    return StreamingResponse(
        content=stream_response(
            client=client,
            model=model,
            msgs=formatted_msgs,
            query=query,
            memory_service=memory_service,
        ),
        media_type="text/event-stream",
    )


@router.post("/api/chat")
async def normal_chat(
    query: Query,
    agent: Agent = Depends(get_agent_dep),
    settings: Settings = Depends(get_settings_dep),
    memory_service: MemoryService = Depends(get_memory_service_dep),
):
    db_settings = await get_settings()
    model = query.model or (
        db_settings.model if db_settings else settings.DEFAULT_MODEL
    )
    thinking = (
        db_settings.settings.get("thinking", False)
        if (db_settings and isinstance(db_settings.settings, dict))
        else settings.THINKING
    )

    run = await asyncio.to_thread(
        agent.run,
        user_input=query.input,
        model=model,
        tool_use=False,
        thinking=thinking,
        chat_history=default_chat_history,
    )
    asyncio.create_task(
        memory_service.memory_pass(
            query.input,
            run["answer"],
            model=model,
        )
    )
    return to_response(run)


@router.post("/api/tool")
async def tool_chat(
    query: Query,
    agent: Agent = Depends(get_agent_dep),
    settings: Settings = Depends(get_settings_dep),
    memory_service: MemoryService = Depends(get_memory_service_dep),
):
    db_settings = await get_settings()
    model = query.model or (
        db_settings.model if db_settings else settings.DEFAULT_MODEL
    )
    thinking = (
        db_settings.settings.get("thinking", False)
        if (db_settings and isinstance(db_settings.settings, dict))
        else settings.THINKING
    )

    run = await asyncio.to_thread(
        agent.run,
        user_input=query.input,
        model=model,
        tool_use=True,
        thinking=thinking,
        chat_history=default_chat_history,
    )
    asyncio.create_task(
        memory_service.memory_pass(
            query.input,
            run["answer"],
            model=model,
        )
    )
    return to_response(run)
