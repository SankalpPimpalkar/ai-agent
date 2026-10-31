from typing import List
from .settings import SETTINGS
from ollama import Message

messages: List[Message] = [Message(role="system", content=SETTINGS.SYSTEM_PROMPT)]