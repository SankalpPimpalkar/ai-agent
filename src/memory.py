from typing import List
# from .schemas import Message
from .settings import SYSTEM_PROMPT
from ollama import Message

messages: List[Message] = [Message(role="system", content=SYSTEM_PROMPT)]