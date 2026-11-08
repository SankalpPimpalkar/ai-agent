from typing import List, Optional
from ollama import Message
from .settings import SETTINGS
from .tools.memory import memory_index, memory_tool
from .tools import memory_tool_spec
import asyncio

messages: List[Message] = [Message(role="system", content=SETTINGS.SYSTEM_PROMPT)]

def _role(m):
    return m["role"] if isinstance(m, dict) else m.role

def build_messages(
    chat_history: Optional[List[Message]] = None,
    tool_use: bool = True,
    system_prompt: Optional[str] = None,
    tool_instructions: Optional[str] = None,
    memory_instructions: Optional[str] = None,
) -> list:
    history = messages if chat_history is None else chat_history

    if system_prompt is not None:
        base_system_prompt = system_prompt
    elif history and _role(history[0]) == "system":
        first_msg = history[0]
        base_system_prompt = (
            first_msg.content
            if hasattr(first_msg, "content")
            else first_msg.get("content", SETTINGS.SYSTEM_PROMPT)
        )
    else:
        base_system_prompt = SETTINGS.SYSTEM_PROMPT

    tool_inst = (
        tool_instructions
        if tool_instructions is not None
        else SETTINGS.TOOL_INSTRUCTIONS
    )
    mem_inst = (
        memory_instructions
        if memory_instructions is not None
        else SETTINGS.MEMORY_INSTRUCTIONS
    )

    if tool_use:
        system_text = "\n\n".join([
            base_system_prompt.strip(),
            tool_inst.strip(),
            mem_inst.strip(),
            f"Your memory files:\n{memory_index()}",
        ])
    else:
        system_text = (
            f"{base_system_prompt.strip()}\n\n"
            "You have no tools in this mode. Do not claim to use tools, "
            "run code, or save or read memory."
        )

    raw_rest = history[1:] if (history and _role(history[0]) == "system") else history[:]
    rest = []
    for m in raw_rest:
        if not tool_use:
            d = m if isinstance(m, dict) else m.model_dump()
            if d.get("role") == "tool":
                continue
            if d.get("tool_calls"):
                if not d.get("content"):
                    continue
                d = {**d, "tool_calls": None}
            rest.append(d)
        else:
            rest.append(m)

    return [{"role": "system", "content": system_text}, *rest]

async def memory_pass(user_msg: str, assistant_msg: str, model: Optional[str] = None):
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
