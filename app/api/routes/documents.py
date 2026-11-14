import json
from typing import List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from langchain_core.vectorstores import VectorStore
from app.api.deps import get_settings, get_vector_store_dep
from app.config import Settings
from app.rag.ingest import (
    delete_document_from_store,
    delete_uploaded_document_record,
    extract_text_from_file,
    get_uploaded_document,
    ingest_document,
    list_uploaded_documents,
)
from app.schemas.chat import (
    DocumentDeleteResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    UploadedDocumentItem,
)

router = APIRouter(tags=["documents"])


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    openapi_extra={
        "requestBody": {
            "description": "Ingest document via raw JSON text or multipart file upload (.txt, .md, .pdf, .docx, etc.)",
            "content": {
                "application/json": {
                    "schema": DocumentIngestRequest.model_json_schema()
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "format": "binary",
                                "description": "Document file to upload (.txt, .md, .pdf, .docx)",
                            },
                            "doc_id": {
                                "type": "string",
                                "description": "Optional custom unique document ID",
                            },
                            "content": {
                                "type": "string",
                                "description": "Optional raw text content if not uploading a file",
                            },
                            "metadata": {
                                "type": "string",
                                "description": "Optional JSON string with metadata key-values",
                            },
                        },
                    }
                },
            },
        }
    },
)
async def ingest_document_route(
    request: Request,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Ingests, chunks, and stores a document in the vector store for RAG.
    Supports both raw text JSON payloads and file uploads in a single route:
    - JSON: `{"content": "...", "doc_id": "...", "metadata": {...}}`
    - File upload (multipart/form-data): file (.txt, .md, .pdf, .docx) + optional doc_id + metadata
    """
    if not settings.ENABLE_RAG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG feature is disabled. Set ENABLE_RAG=true to enable document ingestion.",
        )

    content_type = request.headers.get("content-type", "").lower()
    raw_content = ""
    doc_id: Optional[str] = None
    metadata = {}
    source_label = "raw text"

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON body")

        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object")

        raw_content = body.get("content", "")
        doc_id = body.get("doc_id")
        metadata = body.get("metadata") or {}
        source_label = metadata.get("filename") or metadata.get("source") or "raw text"

    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        upload_file = form.get("file")
        form_content = form.get("content")
        form_doc_id = form.get("doc_id")
        metadata_field = form.get("metadata")

        if form_doc_id and isinstance(form_doc_id, str):
            doc_id = form_doc_id.strip()

        if metadata_field:
            if isinstance(metadata_field, str):
                try:
                    metadata = json.loads(metadata_field)
                except Exception:
                    metadata = {"source": metadata_field}
            elif isinstance(metadata_field, dict):
                metadata = metadata_field

        if upload_file and hasattr(upload_file, "read"):
            filename = upload_file.filename or "uploaded_document"
            try:
                file_bytes = await upload_file.read()
                raw_content = extract_text_from_file(filename, file_bytes)
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err))
            finally:
                if hasattr(upload_file, "close"):
                    await upload_file.close()

            source_label = filename
            metadata.setdefault("filename", filename)
            metadata.setdefault("source", filename)

        elif form_content and isinstance(form_content, str):
            raw_content = form_content
            source_label = metadata.get("filename") or "form text"

    elif "text/plain" in content_type:
        body_bytes = await request.body()
        raw_content = body_bytes.decode("utf-8", errors="replace")
        source_label = "plain text"
        metadata.setdefault("source", "plain_text")

    else:
        # Fallback: attempt JSON first
        try:
            body = await request.json()
            if isinstance(body, dict):
                raw_content = body.get("content", "")
                doc_id = body.get("doc_id")
                metadata = body.get("metadata") or {}
                source_label = metadata.get("filename") or "raw text"
        except Exception:
            pass

    if not raw_content or not raw_content.strip():
        raise HTTPException(
            status_code=422,
            detail="Document content cannot be empty. Provide either 'content' text or a 'file' upload.",
        )

    assigned_doc_id, count = await ingest_document(
        content=raw_content,
        doc_id=doc_id,
        metadata=metadata,
        vector_store=vector_store,
    )

    return DocumentIngestResponse(
        status="success",
        doc_id=assigned_doc_id,
        chunks_count=count,
        message=f"Successfully ingested {count} chunk(s) from '{source_label}' into knowledge base.",
    )


@router.get("/documents", response_model=List[UploadedDocumentItem])
async def list_documents_route(
    limit: int = Query(default=50, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Offset pagination"),
    settings: Settings = Depends(get_settings),
):
    """
    Lists all uploaded documents with their unique IDs, filenames, chunk counts, and timestamps.
    """
    if not settings.ENABLE_RAG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG feature is disabled. Set ENABLE_RAG=true to view documents.",
        )

    records = await list_uploaded_documents(limit=limit, offset=offset)
    return [
        UploadedDocumentItem(
            doc_id=r.doc_id,
            filename=r.filename,
            source=r.source,
            chunks_count=r.chunks_count,
            created_at=r.created_at,
        )
        for r in records
    ]


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document_route(
    doc_id: str,
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store_dep),
):
    """
    Deletes an uploaded document entirely and cleanly from both the SQLite registry
    and the Chroma vector database.
    """
    if not settings.ENABLE_RAG:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAG feature is disabled. Set ENABLE_RAG=true to delete documents.",
        )

    doc_record = await get_uploaded_document(doc_id)
    chunks_count = doc_record.chunks_count if doc_record else None

    # Delete all chunks from the vector store
    await delete_document_from_store(
        doc_id=doc_id,
        chunks_count=chunks_count,
        vector_store=vector_store,
    )

    # Delete the document entry from SQLite database
    if doc_record:
        await delete_uploaded_document_record(doc_id)

    return DocumentDeleteResponse(
        status="success",
        doc_id=doc_id,
        message=f"Successfully deleted document '{doc_id}' and all its chunks from vector database.",
    )
