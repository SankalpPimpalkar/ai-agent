from ollama import Message
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .settings import SETTINGS
from .utils import stream_response, trim_chat, to_response, init, get_settings
from .memory import messages as chat_history, build_messages, memory_pass
from contextlib import asynccontextmanager
from .agent import run_agent
from .schemas import Query
import asyncio

@asynccontextmanager
async def lifespan(app):
    await init()
    yield

app = FastAPI(lifespan=lifespan)
client = SETTINGS.OLLAMA_CLIENT

@app.post("/api/stream")
async def stream_chat(query: Query):
    settings = await get_settings()
    model = query.model if query.model else (settings.model if settings else SETTINGS.DEFAULT_MODEL)
    trim_chat(chat_history)
    chat_history.append(Message(role="user", content=query.input))
    formatted_msgs = build_messages(chat_history=chat_history, tool_use=False)

    return StreamingResponse(
        content=stream_response(client=client, model=model, msgs=formatted_msgs, query=query),
        media_type="text/event-stream",
    )

@app.post("/api/chat")
async def normal_chat(query: Query):
    settings = await get_settings()
    model = query.model if query.model else (settings.model if settings else SETTINGS.DEFAULT_MODEL)
    thinking = (
        settings.settings.get('thinking', False)
        if (settings and isinstance(settings.settings, dict))
        else SETTINGS.THINKING
    )
    run = await asyncio.to_thread(
        run_agent,
        user_input=query.input,
        model=model,
        tool_use=False,
        thinking=thinking,
        chat_history=chat_history,
    )
    asyncio.create_task(memory_pass(query.input, run["answer"], model=model))
    return to_response(run)

@app.post("/api/tool")
async def tool_chat(query: Query):
    settings = await get_settings()
    model = query.model if query.model else (settings.model if settings else SETTINGS.DEFAULT_MODEL)
    thinking = (
        settings.settings.get('thinking', False)
        if (settings and isinstance(settings.settings, dict))
        else SETTINGS.THINKING
    )
    run = await asyncio.to_thread(
        run_agent,
        user_input=query.input,
        model=model,
        tool_use=True,
        thinking=thinking,
        chat_history=chat_history,
    )
    asyncio.create_task(memory_pass(query.input, run["answer"], model=model))
    return to_response(run)
