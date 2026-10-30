from ollama import AsyncClient, ChatResponse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from .tools import tools, functions

SYSTEM_PROMPT = """
You are a friendly and helpful assistant.

Answer only the user's latest message. Use the earlier messages as conversation history for context, but do not answer them again. Keep replies short and simple unless the user asks for detail. Use a warm, polite tone without using any emojis.

Rules:
- If you don't know something, say so instead of guessing.
- Ask a short follow-up question if the request is unclear.
- Don't make up facts, links, or sources.
- Decline harmful or unsafe requests politely.
- Reply in the same language the user writes in.
"""

class Query(BaseModel):
    model: str
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
    return res.message.content

@app.post('/api/tool')
async def tool_chat(query: Query):
    client = AsyncClient()
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))

    while True:
        formatted_msgs = []
        for m in messages:
            formatted_msgs.append(m if isinstance(m, dict) else m.model_dump())
        res: ChatResponse = await client.chat(
            model=query.model, 
            messages=formatted_msgs,
            tools=tools,
            think=True
        )
        messages.append(res.message)

        if res.message.tool_calls:
            for tc in res.message.tool_calls:
                name, args = tc.function.name, tc.function.arguments
                if name in functions:
                    print(f"Calling {name}({args})")
                    res = functions[name](**args)
                    print(f"Tool Result: {str(res)}")
                    messages.append({"role": "tool","content": str(res)})
            continue
        else:
            break
    return res.message.content

@app.post('/api/stream')
def stream_chat(query: Query):
    global messages
    messages = trim_chat(msgs=messages)
    messages.append(Message(role='user', content=query.input))
    formatted_msgs = [m.model_dump() for m in messages]
    return StreamingResponse(
        content=stream_response(model=query.model, msgs=formatted_msgs),
        media_type='text/plain'
    )