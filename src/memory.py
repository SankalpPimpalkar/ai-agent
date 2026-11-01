from typing import List
from ollama import Message
from .settings import SETTINGS, BASE_DIR

MEMORY_FILE = BASE_DIR / "memory.txt"
SUMMARY_TAG = "[CONVERSATION SUMMARY]"

messages: List[Message] = [Message(role="system", content=SETTINGS.SYSTEM_PROMPT)]


def _role(m):
    return m["role"] if isinstance(m, dict) else m.role


def _content(m):
    return (m["content"] if isinstance(m, dict) else m.content) or ""


def _is_summary(m):
    return _role(m) == "system" and _content(m).startswith(SUMMARY_TAG)


def _summary_message(text: str) -> Message:
    return Message(role="system", content=f"{SUMMARY_TAG}\n{text}")


def load_memory():
    if MEMORY_FILE.exists():
        text = MEMORY_FILE.read_text(encoding="utf-8").strip()
        if text:
            messages.insert(1, _summary_message(text))


def _summarize(previous: str, old_chat: List[Message]) -> str:
    chat_text = "\n".join(f"{_role(m)}: {_content(m)[:500]}" for m in old_chat)

    prompt = (
        "Update the memory summary below using the new conversation part.\n"
        "Keep user facts, decisions, preferences, open tasks, and important tool results.\n"
        "Drop small talk. Max 300 words. Return only the summary.\n\n"
        f"CURRENT SUMMARY:\n{previous or '(empty)'}\n\n"
        f"NEW CONVERSATION:\n{chat_text}"
    )
    res = SETTINGS.OLLAMA_CLIENT.chat(
        model=SETTINGS.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return res.message.content.strip()


def summarize(max_length: int = 20):
    system_prompt = messages[0]
    chat_history = messages[1:]
    recent_chat = []
    previous = ""

    if chat_history and _is_summary(chat_history[0]):
        previous = _content(chat_history[0]).removeprefix(SUMMARY_TAG).strip()
        chat_history = chat_history[1:]

    if len(chat_history) < max_length:
        return

    if _role(chat_history[-1]) == "user":
        recent_chat.append(chat_history.pop())

    while chat_history and _role(chat_history[-1]) != "user":
        recent_chat.append(chat_history.pop())

    if chat_history:
        recent_chat.append(chat_history.pop())
    recent_chat.reverse()

    if not chat_history:
        return

    try:
        new_summary = _summarize(previous, chat_history)
    except Exception as e:
        print("Summarize failed:", e)
        return

    MEMORY_FILE.write_text(new_summary, encoding="utf-8")
    messages[:] = [system_prompt, _summary_message(new_summary), *recent_chat]

load_memory()