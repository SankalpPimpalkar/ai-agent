import json
from typing import List, Optional
from ollama import ChatResponse, Message
from .schemas import Trace
from .memory import build_messages, messages as default_messages
from .settings import SETTINGS
from .tools import tools, functions
from .tools.tool_builder import load_custom_tools
from .tools.state import STATE
from .utils import trim_chat

client = SETTINGS.OLLAMA_CLIENT
MAX_STEPS = 25

def run_agent(
    user_input: Optional[str | List[Message]] = None,
    model: Optional[str] = None,
    tool_use: bool = True,
    thinking: bool = False,
    chat_history: Optional[List[Message]] = None,
    system_prompt: Optional[str] = None,
    tool_instructions: Optional[str] = None,
    memory_instructions: Optional[str] = None,
    max_steps: Optional[int] = None,
    **kwargs,
) -> dict:
    if isinstance(user_input, list):
        # 1st argument was chat_history
        passed_history = user_input
        passed_input = model if isinstance(model, str) else kwargs.get("user_input", "")
        passed_model = tool_use if isinstance(tool_use, str) else kwargs.get("model", None)
        passed_tool_use = (
            thinking
            if isinstance(thinking, bool) and "tool_use" not in kwargs
            else kwargs.get("tool_use", True)
        )
        passed_thinking = kwargs.get("thinking", False)

        chat_history = passed_history
        user_input = passed_input
        model = passed_model
        tool_use = passed_tool_use
        thinking = passed_thinking
    else:
        user_input = user_input if user_input is not None else kwargs.get("user_input", "")
        model = model or kwargs.get("model", None)
        tool_use = tool_use if "tool_use" not in kwargs else kwargs["tool_use"]
        thinking = thinking if "thinking" not in kwargs else kwargs["thinking"]

    STATE.reset()
    model = model or SETTINGS.DEFAULT_MODEL
    history = default_messages if chat_history is None else chat_history
    trim_chat(history)
    history.append(Message(role="user", content=user_input))

    trace: List[Trace] = []
    logs: List[str] = [f"User Input: {user_input}"]
    breakers = []
    total_in = total_out = max_step_in = 0
    seen, errors = {}, 0
    answer, steps = "Stopped after too many steps.", 0

    effective_max_steps = max_steps or kwargs.get("max_steps") or MAX_STEPS
    max_iter = effective_max_steps if tool_use else 1

    res = None
    for i in range(max_iter):
        steps = i + 1

        if tool_use:
            custom_tools, custom_funcs = load_custom_tools()
            all_tools = tools + custom_tools
            all_funcs = {**functions, **custom_funcs}
        else:
            all_tools, all_funcs = None, {}

        formatted_messages = build_messages(
            chat_history=history,
            tool_use=tool_use,
            system_prompt=system_prompt,
            tool_instructions=tool_instructions,
            memory_instructions=memory_instructions,
        )

        try:
            res = client.chat(
                model=model,
                messages=formatted_messages,
                tools=all_tools,
                think=thinking,
                options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
            )
        except Exception as e:
            err_str = str(e)
            logs.append(f"[Step {steps}] Ollama error: {err_str}")
            if "XML syntax error" in err_str or "unexpected EOF" in err_str:
                # Retry without tools to get the final answer directly and avoid XML parsing crash
                try:
                    res = client.chat(
                        model=model,
                        messages=[
                            *formatted_messages,
                            Message(
                                role="user",
                                content="Please state the final answer and all requested values directly in text now.",
                            ),
                        ],
                        tools=None,
                        think=False,
                        options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
                    )
                except Exception as e2:
                    logs.append(f"[Step {steps}] Ollama retry failed: {e2}")
                    answer = f"Error during model chat: {e2}"
                    break
            else:
                answer = f"Error during model chat: {err_str}"
                break
        history.append(res.message)

        step_in = res.prompt_eval_count or 0
        total_in += step_in
        total_out += res.eval_count or 0
        max_step_in = max(max_step_in, step_in)

        if res.message.thinking:
            logs.append(f"[Step {steps}] Thinking: {res.message.thinking}")
        if res.message.content:
            logs.append(f"[Step {steps}] Content: {res.message.content}")

        if SETTINGS.DEV_MODE and res.message.thinking:
            print("Thinking:\n", res.message.thinking)
        if SETTINGS.DEV_MODE and res.message.content:
            print("Content:\n", res.message.content)

        if not res.message.tool_calls:
            answer = (res.message.content or "").removeprefix("end_").strip()
            logs.append(f"[Completed] Final answer: {answer}")
            break

        if SETTINGS.DEV_MODE:
            print("Tool Calls:\n", res.message.tool_calls)

        for tc in res.message.tool_calls:
            name, args = tc.function.name, tc.function.arguments
            logs.append(f"[Step {steps}] Tool Call: {name}({args})")

            if name in all_funcs:
                try:
                    result = all_funcs[name](**args)
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Error: unknown tool '{name}'"
            result = str(result)
            logs.append(f"[Step {steps}] Tool Result: {result}")

            errors = errors + 1 if result.startswith("Error") else 0
            if errors >= 4:
                result += "\nToo many errors in a row. Stop and tell the user what is failing."
                breakers.append("errors")
                logs.append(f"[Step {steps}] Breaker triggered: 4 consecutive errors")

            key = (name, json.dumps(args, sort_keys=True, default=str), result)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."
                breakers.append("repeat")
                logs.append(f"[Step {steps}] Breaker triggered: call repeated 3 times with same result")

            if SETTINGS.DEV_MODE:
                print("Tool Result:\n", result)

            trace.append({"name": name, "args": dict(args), "result": result, "step": steps})
            history.append({"role": "tool", "tool_name": name, "content": result})

    if (
        answer == "Stopped after too many steps."
        and res
        and getattr(res, "message", None)
        and res.message.content
    ):
        answer = res.message.content.removeprefix("end_").strip()

    return {
        "answer": answer,
        "steps": steps,
        "trace": trace,
        "logs": logs,
        "breakers": breakers,
        "total_in": total_in,
        "total_out": total_out,
        "total_tokens": total_in + total_out,
        "tokens": total_in + total_out,
        "max_step_prompt_tokens": max_step_in,
    }
