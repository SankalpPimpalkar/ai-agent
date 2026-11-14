from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from app.config import settings


def get_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """Instantiates a ChatOllama chat model instance from configuration."""
    model_name = model or settings.MODEL
    url = base_url or settings.OLLAMA_BASE_URL
    temp = temperature if temperature is not None else settings.TEMPERATURE

    return ChatOllama(
        model=model_name,
        base_url=url,
        temperature=temp,
    )
