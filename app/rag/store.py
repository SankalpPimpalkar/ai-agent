from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from app.config import settings
from app.rag.embeddings import get_embeddings

_GLOBAL_VECTOR_STORE: Optional[VectorStore] = None


def get_vector_store(embeddings: Optional[Embeddings] = None) -> VectorStore:
    """
    Initializes and returns the persistent Chroma vector store.
    Persists embeddings and indexes to settings.CHROMA_PERSIST_DIRECTORY.
    """
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        active_embeddings = embeddings or get_embeddings()
        persist_dir = settings.CHROMA_PERSIST_DIRECTORY
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _GLOBAL_VECTOR_STORE = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=active_embeddings,
            persist_directory=persist_dir,
        )
    return _GLOBAL_VECTOR_STORE


def set_vector_store(store: Optional[VectorStore]) -> None:
    """Explicitly sets or overrides the vector store instance (useful for testing)."""
    global _GLOBAL_VECTOR_STORE
    _GLOBAL_VECTOR_STORE = store


def reset_vector_store() -> None:
    """Resets the vector store instance."""
    global _GLOBAL_VECTOR_STORE
    _GLOBAL_VECTOR_STORE = None
