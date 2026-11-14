from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import settings
from app.rag.store import get_vector_store


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits raw text into overlapping chunks."""
    if not text.strip():
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


async def ingest_document(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    vector_store: Optional[VectorStore] = None,
) -> int:
    """Chunks, embeds, and stores a document in the vector store."""
    store = vector_store or get_vector_store()
    meta = metadata or {}

    chunks = chunk_text(
        content,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    if not chunks:
        return 0

    docs = [
        Document(page_content=chunk, metadata={**meta, "chunk_index": idx})
        for idx, chunk in enumerate(chunks)
    ]
    await store.aadd_documents(docs)
    return len(docs)
