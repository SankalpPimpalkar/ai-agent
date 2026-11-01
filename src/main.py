from ollama import ChatResponse, Message
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .tools import tools, functions
from .tools.tool_builder import load_custom_tools
from .settings import SETTINGS
from .utils import stream_response, trim_chat
from .memory import messages as chat_history, summarize
from .schemas import Query
import asyncio
import json

app = FastAPI()
client = SETTINGS.OLLAMA_CLIENT

@app.post("/api/stream")
async def stream_chat(query: Query):
    await asyncio.to_thread(summarize)
    chat_history.append(Message(role="user", content=query.input))
    formatted_msgs = [m if isinstance(m, dict) else m.model_dump() for m in chat_history]

    return StreamingResponse(
        content=stream_response(client=client, model=query.model, msgs=formatted_msgs),
        media_type="text/event-stream",
    )

@app.post("/api/chat")
async def normal_chat(query: Query):
    await asyncio.to_thread(summarize)
    chat_history.append(Message(role="user", content=query.input))
    formatted_msgs = [m.model_dump() for m in chat_history]

    res: ChatResponse = client.chat(
        model=query.model,
        messages=formatted_msgs,
    )
    chat_history.append(Message(role="assistant", content=res.message.content))

    return {
        "content": res.message.content,
        "usage": {
            "prompt_tokens": res.prompt_eval_count,
            "completion_tokens": res.eval_count,
            "total_tokens": res.prompt_eval_count + res.eval_count,
        },
    }


@app.post("/api/tool")
async def tool_chat(query: Query):
    await asyncio.to_thread(summarize)
    chat_history.append(Message(role="user", content=query.input))

    total_in = total_out = 0
    seen, content, errors  = {}, "Stopped after too many steps.", 0

    for i in range(25):
        custom_tools, custom_funcs = load_custom_tools()
        all_tools = tools + custom_tools
        all_funcs = {**functions, **custom_funcs}
        formatted_msgs = [m if isinstance(m, dict) else m.model_dump() for m in chat_history]

        res: ChatResponse = client.chat(
            model=query.model,
            messages=formatted_msgs,
            tools=all_tools,
            think=SETTINGS.THINKING,
            options={"num_ctx": 8192},
        )
        chat_history.append(res.message)
        total_in += res.prompt_eval_count or 0
        total_out += res.eval_count or 0

        if res.message.thinking:
            print("Thinking:\n", res.message.thinking)

        if res.message.content:
            print("Content:\n", res.message.content)

        if not res.message.tool_calls:
            content = (res.message.content or "").removeprefix("end_").strip()
            break

        if res.message.tool_calls:
            print("Tool Calls:\n", res.message.tool_calls)

        for tc in res.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments

            if name in all_funcs:
                try:
                    result = await asyncio.to_thread(all_funcs[name], **args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Error: unknown tool '{name}'"
            result = str(result)
            errors = errors + 1 if result.startswith("Error") else 0
            
            if errors >= 4:
                result += "\nToo many errors in a row. Stop and tell the user what is failing."

            key = (name, json.dumps(args, sort_keys=True, default=str), result)
            seen[key] = seen.get(key, 0) + 1

            if seen[key] >= 3:
                result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."

            print("Tool Result:\n", result)
            chat_history.append({"role": "tool", "tool_name": name, "content": result})

    return {
        "content": content,
        "usage": {
            "prompt_tokens": total_in,
            "completion_tokens": total_out,
            "total_tokens": total_in + total_out,
        },
    }