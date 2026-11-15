import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, HTTPException, Query, Request, status
from langchain_chroma import Chroma
from sqlalchemy.ext.asyncio import AsyncEngine
from app.db import create_db_engine, init_db
from app.rag.embeddings import get_embeddings
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
    DocumentSearchRequest,
    DocumentSearchResponse,
    RetrievedChunkItem,
    UploadedDocumentItem,
)
from app.schemas.evals import (
    EvalRunRequest,
    EvalRunResponse,
)
from eval.config import (
    EVAL_CHROMA_COLLECTION,
    EVAL_CHROMA_DIR,
    EVAL_DATABASE_URL,
    EVAL_DATA_DIR,
    EVAL_DB_PATH,
    RESULTS_DIR,
)
from eval.runner import RagEvaluator

router = APIRouter(prefix="/evals", tags=["evaluations"])

_EVAL_STORE: Optional[Chroma] = None
_EVAL_ENGINE: Optional[AsyncEngine] = None


def get_eval_vector_store() -> Chroma:
    """Returns the persistent Chroma vector store dedicated to evaluations."""
    global _EVAL_STORE
    if _EVAL_STORE is None:
        Path(EVAL_CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _EVAL_STORE = Chroma(
            collection_name=EVAL_CHROMA_COLLECTION,
            embedding_function=get_embeddings(),
            persist_directory=EVAL_CHROMA_DIR,
        )
    return _EVAL_STORE


def set_eval_vector_store(store: Optional[Chroma]) -> None:
    """Overrides the eval vector store instance (useful for testing)."""
    global _EVAL_STORE
    _EVAL_STORE = store


def reset_eval_vector_store() -> None:
    """Resets the eval vector store instance."""
    global _EVAL_STORE
    _EVAL_STORE = None


def get_eval_engine() -> AsyncEngine:
    """Returns the async database engine dedicated to evaluations."""
    global _EVAL_ENGINE
    if _EVAL_ENGINE is None:
        Path(EVAL_DATA_DIR).mkdir(parents=True, exist_ok=True)
        _EVAL_ENGINE = create_db_engine(EVAL_DATABASE_URL)
    return _EVAL_ENGINE


async def ensure_eval_db() -> None:
    """Ensures evaluation database tables are initialized."""
    eng = get_eval_engine()
    await init_db(eng)


@router.post(
    "/documents",
    response_model=DocumentIngestResponse,
    openapi_extra={
        "requestBody": {
            "description": "Upload document to evaluation knowledge base (raw JSON text or multipart file)",
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
                                "description": "Optional custom document ID",
                            },
                            "content": {
                                "type": "string",
                                "description": "Raw text content",
                            },
                            "metadata": {
                                "type": "string",
                                "description": "JSON string of document metadata",
                            },
                        },
                    }
                },
            },
        }
    },
)
async def upload_eval_document_route(request: Request) -> DocumentIngestResponse:
    """
    Uploads and indexes documents directly into the isolated Evaluation Chroma DB and registry.
    Completely isolated from the main application database and production vector store.
    """
    await ensure_eval_db()
    content_type = request.headers.get("content-type", "").lower()
    raw_content = ""
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
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
        raw_content = body_bytes.decode("utf-8")
        source_label = "plain text payload"

    if not raw_content or not raw_content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document content cannot be empty.",
        )

    assigned_doc_id = doc_id or str(uuid.uuid4())
    store = get_eval_vector_store()
    engine = get_eval_engine()

    doc_id_out, count = await ingest_document(
        content=raw_content,
        doc_id=assigned_doc_id,
        metadata=metadata,
        vector_store=store,
        db_engine=engine,
    )

    return DocumentIngestResponse(
        status="success",
        doc_id=doc_id_out,
        chunks_count=count,
        message=f"Successfully ingested {count} chunk(s) from '{source_label}' into evaluation knowledge base.",
    )


@router.get("/documents", response_model=List[UploadedDocumentItem])
async def list_eval_documents_route(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[UploadedDocumentItem]:
    """Lists all documents currently indexed in the evaluation knowledge base."""
    await ensure_eval_db()
    engine = get_eval_engine()
    records = await list_uploaded_documents(limit=limit, offset=offset, db_engine=engine)
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
async def delete_eval_document_route(doc_id: str) -> DocumentDeleteResponse:
    """Deletes a document entirely from the evaluation Chroma DB and evaluation registry."""
    await ensure_eval_db()
    engine = get_eval_engine()
    store = get_eval_vector_store()

    record = await get_uploaded_document(doc_id, db_engine=engine)
    chunks_count = record.chunks_count if record else None

    await delete_document_from_store(doc_id=doc_id, chunks_count=chunks_count, vector_store=store)
    if record:
        await delete_uploaded_document_record(doc_id, db_engine=engine)

    return DocumentDeleteResponse(
        status="success",
        doc_id=doc_id,
        message=f"Successfully deleted document '{doc_id}' from evaluation knowledge base.",
    )


@router.get("/chunks", response_model=DocumentSearchResponse)
@router.get("/documents/chunks", response_model=DocumentSearchResponse)
async def get_eval_document_chunks_route(
    query: str = Query(..., min_length=1, description="Search query to retrieve matching chunks from eval DB"),
    k: int = Query(default=4, ge=1, le=50, description="Number of chunks to return"),
) -> DocumentSearchResponse:
    """Fetches top-k document chunks matching the query from the evaluation Chroma DB."""
    await ensure_eval_db()
    store = get_eval_vector_store()
    q = query.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query parameter cannot be empty.")

    try:
        if hasattr(store, "asimilarity_search_with_score"):
            results = await store.asimilarity_search_with_score(q, k=k)
        elif hasattr(store, "similarity_search_with_score"):
            results = store.similarity_search_with_score(q, k=k)
        else:
            docs = await store.as_retriever(search_kwargs={"k": k}).ainvoke(q)
            results = [(d, None) for d in docs]
    except Exception:
        docs = await store.as_retriever(search_kwargs={"k": k}).ainvoke(q)
        results = [(d, None) for d in docs]

    chunk_items = [
        RetrievedChunkItem(
            chunk_index=i + 1,
            content=item[0].page_content if isinstance(item, tuple) else item.page_content,
            metadata=item[0].metadata if isinstance(item, tuple) and hasattr(item[0], "metadata") else getattr(item, "metadata", {}),
            score=round(float(item[1]), 4) if isinstance(item, tuple) and item[1] is not None else None,
        )
        for i, item in enumerate(results)
    ]
    return DocumentSearchResponse(query=q, total_chunks=len(chunk_items), chunks=chunk_items)


@router.post("/query", response_model=DocumentSearchResponse)
@router.post("/chunks", response_model=DocumentSearchResponse)
@router.post("/documents/query", response_model=DocumentSearchResponse)
@router.post("/documents/chunks", response_model=DocumentSearchResponse)
async def query_eval_document_chunks_route(
    payload: DocumentSearchRequest,
) -> DocumentSearchResponse:
    """Fetches top-k document chunks for a JSON payload from the evaluation Chroma DB."""
    return await get_eval_document_chunks_route(query=payload.query, k=payload.k)


@router.post("/run", response_model=EvalRunResponse)
async def run_evaluation_route(payload: Optional[EvalRunRequest] = None) -> EvalRunResponse:
    """
    Triggers an evaluation run measuring Correctness, Answer Relevance,
    Groundedness, and Retrieval Relevance. Returns structured results for dashboard panels.
    """
    req = payload or EvalRunRequest()
    yaml_path = req.yaml_path
    clean_on_start = req.clean_on_start

    evaluator = RagEvaluator(
        yaml_path=yaml_path,
        clean_on_start=clean_on_start,
        clean_on_finish=False,
    )
    summary = await evaluator.run()

    return EvalRunResponse(
        status="success",
        timestamp=summary["timestamp"],
        dataset_name=summary["dataset_name"],
        dataset_id=summary.get("dataset_id"),
        total_cases=summary["total_cases"],
        metrics_summary=summary["metrics_summary"],
        cases=summary["cases"],
    )


@router.get("/results")
async def get_latest_eval_results_route() -> Dict[str, Any]:
    """
    Returns the latest evaluation benchmark results and metrics summary
    for dashboard panel consumption without re-running the tests.
    """
    report_file = RESULTS_DIR / "eval_report.json"
    if not report_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation results found. Run evaluations first via POST /evals/run.",
        )

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation report: {exc}")
