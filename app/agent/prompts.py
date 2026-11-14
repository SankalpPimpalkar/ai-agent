from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.config import settings

DEFAULT_SYSTEM_PROMPT = settings.SYSTEM_PROMPT

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ]
)
