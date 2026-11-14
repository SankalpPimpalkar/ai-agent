from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import settings
from app.rag.store import get_vector_store


import io
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import settings
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
