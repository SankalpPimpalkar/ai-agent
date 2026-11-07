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
    trim_chat()
    chat_history.append(Message(role="user", content=query.input))
    formatted_msgs = build_messages(tool_use=False)

    return StreamingResponse(
        content=stream_response(client=client, model=query.model, msgs=formatted_msgs, query=query),
        media_type="text/event-stream",
    )

@app.post("/api/chat")
async def normal_chat(query: Query):
    settings = await get_settings()
    model = query.model if query.model else settings.model
    thinking = settings.settings.get('thinking', False)
    run = await asyncio.to_thread(run_agent, chat_history, query.input, model, thinking)
    # asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)

@app.post("/api/tool")
async def tool_chat(query: Query):
    settings = await get_settings()
    model = query.model if query.model else settings.model
    thinking = settings.settings.get('thinking', False)
    run = await asyncio.to_thread(run_agent, query.input, model, thinking)
    # asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)