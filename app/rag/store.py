from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from app.rag.embeddings import get_embeddings

_GLOBAL_VECTOR_STORE: Optional[VectorStore] = None


def get_vector_store(embeddings: Optional[Embeddings] = None) -> VectorStore:
    """
    Initializes and returns the shared vector store.
    Uses InMemoryVectorStore as default, easily swappable for Chroma or FAISS.
    """
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        active_embeddings = embeddings or get_embeddings()
        _GLOBAL_VECTOR_STORE = InMemoryVectorStore(active_embeddings)
    return _GLOBAL_VECTOR_STORE


def reset_vector_store() -> None:
    """Resets the vector store instance (useful for testing)."""
    global _GLOBAL_VECTOR_STORE
    _GLOBAL_VECTOR_STORE = None
