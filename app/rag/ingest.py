from datetime import datetime, timezone
import io
import os
from typing import Any, Dict, List, Optional
import uuid
import xml.etree.ElementTree as ET
import zipfile
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import settings
from app.models.document import UploadedDocument
from app.rag.store import get_vector_store


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """
    Extracts text content from uploaded files:
    - Text and code files (.txt, .md, .csv, .json, .html, .py, etc.)
    - PDF documents (.pdf) via pypdf
    - Word documents (.docx) via standard zipfile / XML parsing
    """
    if not file_bytes:
        return ""

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF '{filename}': {e}") from e

    elif ext == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as docx:
                xml_content = docx.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for p in tree.iterfind(".//w:p", ns):
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    paragraphs.append("".join(texts))
            return "\n\n".join(paragraphs)
        except Exception as e:
            raise ValueError(f"Failed to extract text from Word document '{filename}': {e}") from e

    else:
        # Default plain text / markdown / code files
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Could not decode text from '{filename}': {e}") from e


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
    doc_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    vector_store: Optional[VectorStore] = None,
) -> tuple[str, int]:
    """
    Chunks, embeds, and stores a document in the vector store tagged with a unique doc_id.
    Returns (doc_id, chunks_count).
    """
    store = vector_store or get_vector_store()
    active_doc_id = doc_id or str(uuid.uuid4())
    meta = metadata or {}

    chunks = chunk_text(
        content,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    if not chunks:
        return active_doc_id, 0

    docs = [
        Document(
            page_content=chunk,
            metadata={
                **meta,
                "doc_id": active_doc_id,
                "chunk_index": idx,
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
    chunk_ids = [f"{active_doc_id}_{idx}" for idx in range(len(chunks))]

    await store.aadd_documents(docs, ids=chunk_ids)

    # Record document in SQLite database
    try:
        filename = meta.get("filename") or "document.txt"
        source = meta.get("source") or filename
        await record_uploaded_document(
            doc_id=active_doc_id,
            filename=filename,
            source=source,
            chunks_count=len(docs),
        )
    except Exception:
        pass

    return active_doc_id, len(docs)


async def delete_document_from_store(
    doc_id: str,
    chunks_count: Optional[int] = None,
    vector_store: Optional[VectorStore] = None,
) -> None:
    """
    Deletes all chunks belonging to doc_id entirely and cleanly from the vector store.
    """
    store = vector_store or get_vector_store()

    # 1. Try Chroma where filter
    deleted_by_filter = False
    try:
        if hasattr(store, "adelete"):
            await store.adelete(where={"doc_id": doc_id})
        elif hasattr(store, "delete"):
            store.delete(where={"doc_id": doc_id})
        deleted_by_filter = True
    except Exception:
        pass

    # 2. Also ensure deletion by chunk IDs
    count = chunks_count or 100
    chunk_ids = [f"{doc_id}_{idx}" for idx in range(count)]
    try:
        if hasattr(store, "adelete"):
            await store.adelete(ids=chunk_ids)
        elif hasattr(store, "delete"):
            store.delete(ids=chunk_ids)
    except Exception:
        try:
            if hasattr(store, "adelete"):
                await store.adelete(chunk_ids)
            elif hasattr(store, "delete"):
                store.delete(chunk_ids)
        except Exception:
            pass


async def record_uploaded_document(
    doc_id: str,
    filename: str,
    source: str,
    chunks_count: int,
) -> Optional[UploadedDocument]:
    """Records an uploaded document in the database registry."""
    from app.db import engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    now = datetime.now(timezone.utc)
    async with async_session() as session:
        statement = select(UploadedDocument).where(UploadedDocument.doc_id == doc_id)
        result = await session.exec(statement)
        doc = result.first()
        if doc is None:
            doc = UploadedDocument(
                doc_id=doc_id,
                filename=filename,
                source=source,
                chunks_count=chunks_count,
                created_at=now,
            )
            session.add(doc)
        else:
            doc.filename = filename
            doc.source = source
            doc.chunks_count = chunks_count
        await session.commit()
        return doc


async def get_uploaded_document(doc_id: str) -> Optional[UploadedDocument]:
    """Retrieves an uploaded document record from SQLite."""
    from app.db import engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(UploadedDocument).where(UploadedDocument.doc_id == doc_id)
        result = await session.exec(statement)
        return result.first()


async def list_uploaded_documents(limit: int = 50, offset: int = 0) -> List[UploadedDocument]:
    """Lists uploaded document records from SQLite ordered by creation time."""
    from app.db import engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = (
            select(UploadedDocument)
            .order_by(UploadedDocument.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        results = await session.exec(statement)
        return list(results.all())


async def delete_uploaded_document_record(doc_id: str) -> Optional[UploadedDocument]:
    """Deletes an uploaded document record from SQLite."""
    from app.db import engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(UploadedDocument).where(UploadedDocument.doc_id == doc_id)
        result = await session.exec(statement)
        doc = result.first()
        if doc:
            await session.delete(doc)
            await session.commit()
            return doc
        return None
