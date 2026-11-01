from typing import List
from ollama import Message
from .settings import SETTINGS
from .tools.memory import memory_index, memory_tool
from .tools import memory_tool_spec
import asyncio

messages: List[Message] = [Message(role="system", content=SETTINGS.SYSTEM_PROMPT)]

def build_messages():
    global messages

    system = {
        'role': 'system',
        'content': f"{SETTINGS.SYSTEM_PROMPT}\n\nYour Memory Files:\n{memory_index()}"
    }
    rest = [m if isinstance(m, dict) else m.model_dump() for m in messages[1:]]
    return [system] + rest

async def memory_pass(user_msg: str, assistant_msg: str):
    prompt = (
        "You manage the agent's long-term memory.\n"
        f"Current memory files:\n{memory_index()}\n\n"
        "Does this exchange contain anything worth saving (personal info, preferences, "
        "decisions, what the user is learning, topics asked)? "
        "If yes, use memory_tool (update an existing file if one fits). "
        "If not, reply NONE.\n\n"
        f"User: {user_msg}\nAssistant: {assistant_msg}"
    )
    msgs: List[Message] = [{"role": "user", "content": prompt}]

    for _ in range(5):
        res = await asyncio.to_thread(
            SETTINGS.OLLAMA_CLIENT.chat,
            model=SETTINGS.DEFAULT_MODEL,
            messages=msgs,
            tools=[memory_tool_spec],
        )
        msgs.append(res.message)
        if not res.message.tool_calls:
            return
        for call in res.message.tool_calls:
            result = memory_tool(**call.function.arguments)
            msgs.append({"role": "tool", "content": result})