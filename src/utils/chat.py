import asyncio
from typing import Any, List, Optional
from ollama import ChatResponse, Message
from src.schemas import Query


def _role(m: Any) -> str:
    return m["role"] if isinstance(m, dict) else m.role


async def stream_response(client: Any, model: str, msgs: List[Any], query: Query):
    try:
        from src.memory import messages, memory_pass

        stream: ChatResponse = client.chat(
            model=model, messages=msgs, stream=True
        )
        full_response = ""

        for chunk in stream:
            out = chunk["message"]["content"]
            full_response += str(out)
            yield out

        messages.append(Message(role="assistant", content=full_response))
        asyncio.create_task(memory_pass(query.input, full_response, model=model))
    except Exception as e:
        yield f"\n[Error: {str(e)}]"


def trim_chat(chat_history: Optional[List[Message]] = None, max_messages: int = 20):
    if chat_history is None:
        from src.memory import messages
        history = messages
    else:
        history = chat_history

    if not history:
        return
    has_system = _role(history[0]) == "system"
    system_prompt = history[0] if has_system else None
    chat = history[1:] if has_system else history[:]

    if len(chat) > max_messages:
        chat = chat[-max_messages:]
        while chat and _role(chat[0]) != "user":
            chat.pop(0)

    history[:] = ([system_prompt] if system_prompt else []) + chat


def to_response(run: dict) -> dict:
    total_tokens = run.get("total_tokens", run.get("tokens", 0))
    return {
        "content": run.get("answer", ""),
        "usage": {
            "prompt_tokens": run.get("total_in", 0),
            "completion_tokens": run.get("total_out", 0),
            "total_tokens": total_tokens,
        },
    }
