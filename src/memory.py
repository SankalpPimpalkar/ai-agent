from typing import List
from ollama import Message
from .settings import SETTINGS
from .tools.memory import memory_index, memory_tool
from .tools import memory_tool_spec
import asyncio

messages: List[Message] = [Message(role="system", content=SETTINGS.SYSTEM_PROMPT)]

def build_messages(chat_history: List[Message]) -> list:
    system_text = "\n\n".join([
        SETTINGS.SYSTEM_PROMPT,
        SETTINGS.TOOL_INSTRUCTIONS,
        SETTINGS.MEMORY_INSTRUCTIONS,
        f"Your memory files:\n{memory_index()}",
    ])
    rest = [m if isinstance(m, dict) else m.model_dump() for m in chat_history[1:]]
    return [{"role": "system", "content": system_text}, *rest]

async def memory_pass(user_msg: str, assistant_msg: str, model: str | None = None):
    system = "\n\n".join([
        SETTINGS.MEMORY_INSTRUCTIONS,
        f"Current memory files:\n{memory_index()}",
    ])
    exchange = (
        "Does this exchange contain anything worth saving? "
        "Read the target file first. If the fact is already saved, reply NONE. "
        "Otherwise use memory_tool, updating an existing line when one fits. "
        "If nothing is worth saving, reply NONE.\n\n"
        f"User: {user_msg}\nAssistant: {assistant_msg}"
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": exchange},
    ]

    try:
        for _ in range(5):
            res = await asyncio.to_thread(
                SETTINGS.OLLAMA_CLIENT.chat,
                model=model or SETTINGS.DEFAULT_MODEL,
                messages=msgs,
                tools=[memory_tool_spec],
                think=False,
                options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
            )
            msgs.append(res.message)
            if not res.message.tool_calls:
                return
            for call in res.message.tool_calls:
                name = call.function.name
                result = (
                    memory_tool(**call.function.arguments)
                    if name == "memory_tool"
                    else f"Error: unknown tool '{name}'"
                )
                msgs.append({"role": "tool", "tool_name": name, "content": str(result)})
    except Exception as e:
        print("memory_pass failed:", e)