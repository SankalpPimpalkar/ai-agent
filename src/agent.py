import json
from typing import List
from ollama import ChatResponse, Message
from evals.schemas import Trace
from .memory import build_messages
from .settings import SETTINGS
from .tools import tools, functions
from .tools.tool_builder import load_custom_tools
from .utils import trim_chat

client = SETTINGS.OLLAMA_CLIENT
MAX_STEPS = 25

def run_agent(
    chat_history: List[Message],
    user_input: str,
    model: str | None = None,
    thinking: bool = False,
) -> dict:
    model = model or SETTINGS.DEFAULT_MODEL
    trim_chat()
    chat_history.append(Message(role="user", content=user_input))

    trace: List[Trace] = []
    breakers = []
    total_in = total_out = max_step_in = 0
    seen, errors = {}, 0
    answer, steps = "Stopped after too many steps.", 0

    for i in range(MAX_STEPS):
        steps = i + 1
        custom_tools, custom_funcs = load_custom_tools()
        all_tools = tools + custom_tools
        all_funcs = {**functions, **custom_funcs}

        res: ChatResponse = client.chat(
            model=model,
            messages=chat_history,
            tools=all_tools,
            think=thinking,
            options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
        )
        chat_history.append(res.message)

        step_in = res.prompt_eval_count or 0
        total_in += step_in
        total_out += res.eval_count or 0
        max_step_in = max(max_step_in, step_in)

        if SETTINGS.DEV_MODE and res.message.thinking:
            print("Thinking:\n", res.message.thinking)
        if SETTINGS.DEV_MODE and res.message.content:
            print("Content:\n", res.message.content)

        if not res.message.tool_calls:
            answer = (res.message.content or "").removeprefix("end_").strip()
            break

        if SETTINGS.DEV_MODE:
            print("Tool Calls:\n", res.message.tool_calls)

        for tc in res.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments

            if name in all_funcs:
                try:
                    result = all_funcs[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Error: unknown tool '{name}'"
            result = str(result)

            errors = errors + 1 if result.startswith("Error") else 0
            if errors >= 4:
                result += "\nToo many errors in a row. Stop and tell the user what is failing."
                breakers.append("errors")

            key = (name, json.dumps(args, sort_keys=True, default=str), result)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."
                breakers.append("repeat")

            if SETTINGS.DEV_MODE:
                print("Tool Result:\n", result)

            trace.append({"name": name, "args": dict(args), "result": result})
            chat_history.append({"role": "tool", "tool_name": name, "content": result})

    return {
        "answer": answer,
        "steps": steps,
        "trace": trace,
        "breakers": breakers,
        "total_in": total_in,
        "total_out": total_out,
        "tokens": total_in + total_out,
        "max_step_prompt_tokens": max_step_in,
    }