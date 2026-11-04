from ollama import Message
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .settings import SETTINGS
from .utils import stream_response, trim_chat, to_response
from .memory import messages as chat_history, build_messages, memory_pass
from .agent import run_agent
from .schemas import Query
import asyncio

app = FastAPI()
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
    run = await asyncio.to_thread(run_agent, query.input, query.model, False)
    asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)

@app.post("/api/tool")
async def tool_chat(query: Query):
    run = await asyncio.to_thread(run_agent, query.input, query.model, True)
    asyncio.create_task(memory_pass(query.input, run["answer"]))
    return to_response(run)