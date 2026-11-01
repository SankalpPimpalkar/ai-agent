from typing import List
from .schemas import Query
from ollama import AsyncClient, ChatResponse, Message
from .memory import messages, memory_pass
import asyncio

async def stream_response(client: AsyncClient, model: str, msgs: List[Message], query: Query):
    try:
        stream: ChatResponse = client.chat(
            model=model, messages=msgs, stream=True
        )
        full_response = ""

        async for chunk in stream:
            out = chunk["message"]["content"]
            full_response += str(out)
            yield out

        messages.append(Message(role="assistant", content=full_response))
        asyncio.create_task(memory_pass(query.input, full_response))
    except Exception as e:
        yield f"\n[Error: {str(e)}]"

def _role(m):
    return m["role"] if isinstance(m, dict) else m.role

def trim_chat(max_messages: int = 20):
    system_prompt = messages[0]
    chat = messages[1:]
    if len(chat) > max_messages:
        chat = chat[-max_messages:]
        while chat and _role(chat[0]) != "user":
            chat.pop(0)
    messages[:] = [system_prompt] + chat