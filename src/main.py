from ollama import AsyncClient, ChatResponse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from .tools import tools, functions
from .tools.tool_builder import load_custom_tools
from .settings import DEFAULT_MODEL, SYSTEM_PROMPT, THINKING_MODEL
class Query(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    input: str

class Message(BaseModel):
    role: str
    content: str

async def stream_response(model: str, msgs: List[Message]):
    try:
        global messages
        client = AsyncClient()
        stream: ChatResponse = await client.chat(
            model=model, 
            messages=msgs, 
            stream=True
        )
        full_response = ""

        async for chunk in stream:
            out = chunk['message']['content']
            full_response += str(out)
            yield out

        messages.append(Message(role='assistant', content=full_response))
    except Exception as e:
        yield f"\n[Error: {str(e)}]"

def trim_chat(msgs: List[Message], max_turns: int = 20):
    system_prompt = msgs[0]
    chat_turns = msgs[1:]
    max_messages = max_turns

    if len(chat_turns) > max_messages:
        chat_turns = chat_turns[-max_messages:]
    return [system_prompt] + chat_turns

app = FastAPI()
messages: List[Message] = [Message(role = 'system', content = SYSTEM_PROMPT)]

@app.post('/api/chat')
async def normal_chat(query: Query):
    client = AsyncClient()
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))
    formatted_msgs = [m.model_dump() for m in messages]
    res: ChatResponse = await client.chat(
        model=query.model, 
        messages=formatted_msgs,
    )
    messages.append(Message(role='assistant', content=res.message.content))
    return {
        "content": res.message.content,
        "usage": {
            "prompt_tokens": res.prompt_eval_count,
            "completion_tokens": res.eval_count,
            "total_tokens": res.prompt_eval_count + res.eval_count
        }
    }

@app.post('/api/tool')
async def tool_chat(query: Query):
    client = AsyncClient()
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(15):
        custom_tools, custom_funcs = load_custom_tools()
        all_tools = tools + custom_tools
        all_funcs = {**functions, **custom_funcs}
        formatted_msgs = []

        for m in messages:
            formatted_msgs.append(m if isinstance(m, dict) else m.model_dump())

        res: ChatResponse = await client.chat(
            model=query.model, 
            messages=formatted_msgs,
            tools=all_tools,
            think=THINKING_MODEL
        )
        messages.append(res.message)
        total_input_tokens += res.prompt_eval_count
        total_output_tokens += res.eval_count

        if res.message.tool_calls:
            for tc in res.message.tool_calls:
                name, args = tc.function.name, tc.function.arguments
                if name in all_funcs:
                    print(f"Calling {name}({args})\n")
                    res = all_funcs[name](**args)
                    # print(f"Tool Result: {str(res)}")
                else:
                    res = f"Error: unknown tool '{name}'"
                messages.append({"role": "tool","content": str(res)})
            continue
        else:
            break
    
    content = (res.message.content or "").removeprefix("end_").strip()

    return {
        "content": content,
        "usage": {
            "prompt_tokens": total_input_tokens,
            "completion_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens
        }
    }

@app.post('/api/stream')
def stream_chat(query: Query):
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))
    formatted_msgs = [m.model_dump() for m in messages]
    return StreamingResponse(
        content=stream_response(model=query.model, msgs=formatted_msgs),
        media_type='text/event-stream'
    )