from typing import Any, Iterable, List, Optional, Union
from ollama import Message


def _role(m: Any) -> str:
    return m["role"] if isinstance(m, dict) else m.role


def trim_chat(
    chat_history: Optional[List[Union[Message, dict]]] = None,
    max_messages: int = 20,
) -> None:
    """Trims old messages preserving system instructions and window bounds."""
    if chat_history is None:
        return
    history = chat_history
    if not history:
        return
    has_system = _role(history[0]) == "system"
    system_prompt = history[0] if has_system else None
    chat = history[1:] if has_system else history[:]

    if len(chat) > max_messages:
        chat = chat[-max_messages:]
        while chat and _role(chat[0]) != "user":
            chat.pop(0)

    history[:] = ([system_prompt] if system_prompt else []) + chat


class ConversationHistory:
    """
    Manages chat messages for an agent execution.
    Encapsulates history storage, appending, and window trimming.
    """

    def __init__(self, initial_messages: Optional[List[Union[Message, dict]]] = None):
        self._messages: List[Union[Message, dict]] = []
        if initial_messages is not None:
            self._messages = list(initial_messages)

    def append(self, message: Union[Message, dict]) -> None:
        self._messages.append(message)

    def extend(self, messages: Iterable[Union[Message, dict]]) -> None:
        self._messages.extend(messages)

    def trim(self, max_messages: int = 20) -> None:
        trim_chat(self._messages, max_messages=max_messages)

    def get_messages(self) -> List[Union[Message, dict]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __getitem__(self, index: Any) -> Any:
        return self._messages[index]
