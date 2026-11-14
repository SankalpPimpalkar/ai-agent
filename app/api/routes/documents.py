from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.vectorstores import VectorStore
from app.api.deps import get_settings, get_vector_store_dep
from app.config import Settings
from app.rag.ingest import ingest_document
from app.schemas.chat import DocumentIngestRequest, DocumentIngestResponse

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentIngestResponse)
async def ingest_document_route(
    request: DocumentIngestRequest,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Ingests, chunks, and stores a document in the vector store for RAG.
    Only active if ENABLE_RAG is set to True.
    """
    if not settings.ENABLE_RAG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG feature is disabled. Set ENABLE_RAG=true to enable document ingestion.",
        )

    if not request.content.strip():
        raise HTTPException(status_code=422, detail="Document content cannot be empty")

    count = await ingest_document(
        content=request.content,
        metadata=request.metadata,
        vector_store=vector_store,
    )

    return DocumentIngestResponse(
        status="success",
        chunks_count=count,
        message=f"Successfully ingested {count} chunk(s) into knowledge base.",
    )
