from typing import List, Optional
from ollama import Client, ChatResponse, Message
from src.agent.settings import SETTINGS


def chat_step(
    client: Client,
    model: str,
    messages: List[Message],
    tools: Optional[list],
    thinking: bool,
    logs: List[str],
    step: int,
) -> Optional[ChatResponse]:
    """
    Executes a single model turn via Ollama with error recovery for XML syntax/EOF errors.
    """
    try:
        return client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=thinking,
            options={"num_ctx": SETTINGS.MODEL_CTX_SZ},
        )
    except Exception as e:
        err_str = str(e)
        logs.append(f"[Step {step}] Ollama error: {err_str}")

        if "XML syntax error" in err_str or "unexpected EOF" in err_str:
            # Retry without tools so unescaped HTML/math tokens do not crash Ollama's Go XML parser
            try:
                logs.append(
                    f"[Step {step}] Retrying model without tools to obtain text response..."
                )
                return client.chat(
                    model=model,
                    messages=[
                        *messages,
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
                logs.append(f"[Step {step}] Ollama retry failed: {e2}")
                return None
        return None
