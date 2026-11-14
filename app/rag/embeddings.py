from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from app.config import settings


def get_embeddings(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Embeddings:
    """Instantiates an OllamaEmbeddings instance."""
    embedding_model = model or settings.EMBEDDING_MODEL
    url = base_url or settings.OLLAMA_BASE_URL
    return OllamaEmbeddings(model=embedding_model, base_url=url)
