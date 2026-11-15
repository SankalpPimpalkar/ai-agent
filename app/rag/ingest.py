from datetime import datetime, timezone
import io
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid
import xml.etree.ElementTree as ET
import zipfile
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from app.config import settings
from app.models.document import UploadedDocument
from app.rag.store import get_vector_store

logger = logging.getLogger("rag_ingest")


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """
    Extracts text content from uploaded files:
    - Text and code files (.txt, .md, .csv, .json, .html, .py, etc.)
    - PDF documents (.pdf) via pymupdf4llm (Markdown + tables), pdfplumber, or pypdf
    - Word documents (.docx) via standard zipfile / XML parsing with heading detection
    """
    if not file_bytes:
        return ""

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        # 1. Primary: pymupdf4llm preserves full layout, markdown section headers, and tables
        try:
            import pymupdf
            import pymupdf4llm

            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            md_text = pymupdf4llm.to_markdown(doc)
            if md_text is not None and md_text.strip():
                return md_text.strip()
            elif md_text is not None and len(doc) == 0:
                return ""
        except Exception as e:
            logger.warning(
                "pymupdf4llm extraction failed for '%s': %s. Trying pdfplumber fallback.",
                filename,
                e,
            )

        # 2. Secondary: pdfplumber for table extraction and structured page layout
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        table_blocks = []
                        for table in tables:
                            rows = []
                            for row in table:
                                if row and any(cell is not None and str(cell).strip() for cell in row):
                                    rows.append("| " + " | ".join(str(cell or "").strip() for cell in row) + " |")
                            if rows:
                                table_blocks.append("\n".join(rows))
                        if table_blocks:
                            page_text += "\n\n" + "\n\n".join(table_blocks)
                    if page_text.strip():
                        pages_text.append(page_text.strip())
                if pages_text:
                    return "\n\n".join(pages_text)
        except Exception as e:
            logger.warning(
                "pdfplumber extraction failed for '%s': %s. Trying pypdf fallback.",
                filename,
                e,
            )

        # 3. Fallback: pypdf basic text extraction
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p.strip() for p in pages if p.strip())
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
                # Detect heading styles if present
                style_node = p.find(".//w:pPr/w:pStyle", ns)
                style_val = (
                    style_node.attrib.get(f"{{{ns['w']}}}val", "")
                    if style_node is not None
                    else ""
                )
                texts = [node.text for node in p.iterfind(".//w:t", ns) if node.text]
                if texts:
                    para_text = "".join(texts).strip()
                    if not para_text:
                        continue
                    if "Heading1" in style_val:
                        paragraphs.append(f"# {para_text}")
                    elif "Heading2" in style_val:
                        paragraphs.append(f"## {para_text}")
                    elif "Heading3" in style_val:
                        paragraphs.append(f"### {para_text}")
                    else:
                        paragraphs.append(para_text)
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
    """
    Splits text along section headings and prepends the heading hierarchy to every chunk
    so that retrieved chunks preserve their exact semantic context and headings.
    Falls back gracefully to sliding-window / paragraph chunking if no headings are present.
    """
    if not text or not text.strip():
        return []

    return chunk_text_by_sections(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_subsections(text: str, header: str) -> List[str]:
    """
    Splits text along bold subsection lead-ins (e.g. '**12.3 ShopNest Pay.**', '**Nevada.**')
    before any character-based splitting, prepending the section header to each part.
    Preserves bold phrases ending with a colon (e.g. '**Breach notification:**') inside their section.
    """
    parts = re.split(r'(?=\*\*[^*]{2,80}\.\*\*)', text)
    return [f"{header}\n\n{p.strip()}" if header else p.strip() for p in parts if p.strip()]


def linearize_markdown_table(table_text: str) -> str:
    """
    Converts markdown tables into clean key-value text following the pattern:
    'Customer support chats: retained 3 years. Call recordings: 12 months'
    or 'Question: Answer'. Removes duplicate headers and empty divider rows.
    """
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return table_text

    header_row = None
    cleaned_rows = []

    for line in lines:
        if not (line.startswith("|") and line.endswith("|")):
            continue
        if re.match(r"^\|?\s*[-:]+[-| :]*\|?$", line):
            continue

        cells = [c.strip().replace("<br>", " ") for c in line.split("|")[1:-1]]
        if not any(cells):
            continue

        if header_row is None:
            header_row = cells
            continue

        if cells == header_row:
            continue

        cleaned_rows.append(cells)

    if not header_row or not cleaned_rows:
        return table_text

    output_lines = []
    h_lower = [h.lower() for h in header_row]

    for row in cleaned_rows:
        if len(row) < len(header_row):
            row = row + [""] * (len(header_row) - len(row))

        if len(header_row) == 2:
            q_val = row[0]
            a_val = row[1]
            if not q_val and not a_val:
                continue
            if "retention" in h_lower[1] and not a_val.lower().startswith("retained"):
                output_lines.append(f"{q_val}: retained {a_val}")
            else:
                output_lines.append(f"{q_val}: {a_val}")
        elif any("retention" in h for h in h_lower):
            item_name = row[0]
            ret_idx = next(i for i, h in enumerate(h_lower) if "retention" in h)
            ret_val = row[ret_idx]
            other_parts = [
                row[idx]
                if (":" in row[idx] or header_row[idx].lower() == "notes")
                else f"{header_row[idx]}: {row[idx]}"
                for idx in range(1, len(row))
                if idx != ret_idx and row[idx]
            ]
            if (
                ret_val
                and not ret_val.lower().startswith("retained")
                and not ret_val.lower().startswith("life")
                and not ret_val.lower().startswith("until")
            ):
                ret_str = f"retained {ret_val}"
            else:
                ret_str = ret_val

            line_str = f"{item_name}: {ret_str}" if ret_str else item_name
            if other_parts:
                line_str += ". " + ". ".join(other_parts)
            output_lines.append(line_str)
        else:
            item_name = row[0]
            other_parts = [
                row[idx] if ":" in row[idx] else f"{header_row[idx]}: {row[idx]}"
                for idx in range(1, len(row))
                if row[idx]
            ]
            line_str = item_name
            if other_parts:
                line_str += ". " + ". ".join(other_parts)
            output_lines.append(line_str)

    return "\n".join(output_lines)


def linearize_markdown_table_blocks(text: str) -> str:
    """Finds markdown table blocks within text (including multiline and blank-line separated) and linearizes them."""
    lines = text.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and line.endswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt.startswith("|") and nxt.endswith("|"):
                    table_lines.append(nxt)
                    i += 1
                elif not nxt:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and lines[j].strip().startswith("|") and lines[j].strip().endswith("|"):
                        i = j
                    else:
                        break
                else:
                    break
            lin = linearize_markdown_table("\n".join(table_lines))
            new_lines.append(lin)
        else:
            new_lines.append(lines[i])
            i += 1
    return "\n".join(new_lines)


def chunk_text_by_sections(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Splits text into section-aware chunks prepended with their section heading breadcrumbs.
    Splits on bold subsection markers before any character-based splitting.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # 1. Detect Markdown headings (# through ######)
    md_heading_pattern = re.compile(r"^(#{1,6})\s+([^\n]+)$", re.MULTILINE)
    matches = list(md_heading_pattern.finditer(stripped))

    # 2. If no Markdown headings, check for numbered/formal section titles
    if not matches:
        text_heading_pattern = re.compile(
            r"^((?:SECTION|ARTICLE|CHAPTER)\s+[0-9A-Z]+[:.]?|[0-9]+(?:\.[0-9]+)*[:.]?)\s+([A-Z][^\n]+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = list(text_heading_pattern.finditer(stripped))

    # 3. If still no headings detected, fallback to paragraph/sliding-window chunking
    if not matches:
        return _fallback_chunking(stripped, chunk_size, chunk_overlap)

    # 4. Extract sections with heading hierarchy
    sections: List[Tuple[str, str]] = []
    first_match = matches[0]
    if first_match.start() > 0:
        pre_content = stripped[:first_match.start()].strip()
        if pre_content:
            sections.append(("", pre_content))

    breadcrumbs: Dict[int, str] = {}
    is_markdown = hasattr(matches[0], "group") and matches[0].group(1).startswith("#")

    for i, match in enumerate(matches):
        if is_markdown:
            level = len(match.group(1))
            title = match.group(2).strip()
        else:
            sec_num = match.group(1).strip()
            sec_title = match.group(2).strip()
            level = sec_num.count(".") + 1
            title = f"{sec_num} {sec_title}"

        # Clean trailing formatting or symbols
        title = re.sub(r"[\s*#_]+$", "", title).strip()

        # Update hierarchy: clear deeper or equal levels
        breadcrumbs = {lvl: t for lvl, t in breadcrumbs.items() if lvl < level}
        breadcrumbs[level] = title

        path = " > ".join(breadcrumbs[lvl] for lvl in sorted(breadcrumbs.keys()))

        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(stripped)
        content = stripped[content_start:content_end].strip()

        sections.append((path, content))

    # 5. Build chunks, prepending heading path to each chunk
    chunks: List[str] = []
    for path, content in sections:
        header = f"## Section: {path}" if path else ""
        if not content:
            if header:
                chunks.append(header)
            continue

        prefix = f"{header}\n\n" if header else ""
        linearized_content = linearize_markdown_table_blocks(content)
        subsections = split_subsections(linearized_content, header)
        for sub_chunk in subsections:
            if len(sub_chunk) <= chunk_size:
                chunks.append(sub_chunk.strip())
            else:
                body = (
                    sub_chunk[len(prefix) :].strip()
                    if sub_chunk.startswith(prefix)
                    else sub_chunk.strip()
                )
                sub_chunks = _split_section_content(body, prefix, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)

    return chunks


def _split_section_content(
    content: str,
    prefix: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Splits section content exceeding chunk_size, prepending prefix to each chunk."""
    prefix_len = len(prefix)
    max_body_size = max(100, chunk_size - prefix_len)
    step = max(1, max_body_size - chunk_overlap)

    # First attempt splitting by paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content]

    result: List[str] = []
    current_paras: List[str] = []
    current_len = 0

    for p in paragraphs:
        # Check if paragraph is a markdown table
        if p.startswith("|") and "\n|" in p and len(p) > max_body_size:
            # Flush existing paragraphs before table
            if current_paras:
                body = "\n\n".join(current_paras)
                result.append(f"{prefix}{body}".strip())
                current_paras = []
                current_len = 0
            # Split table rows while keeping table header row
            table_chunks = _split_table_block(p, max_body_size)
            for tch in table_chunks:
                result.append(f"{prefix}{tch}".strip())
            continue

        p_len = len(p)
        if current_len + p_len + (2 if current_paras else 0) <= max_body_size:
            current_paras.append(p)
            current_len += p_len + (2 if len(current_paras) > 1 else 0)
        else:
            if current_paras:
                body = "\n\n".join(current_paras)
                result.append(f"{prefix}{body}".strip())
                last = current_paras[-1]
                if len(last) <= chunk_overlap:
                    current_paras = [last, p] if len(last) + len(p) + 2 <= max_body_size else [p]
                    current_len = sum(len(x) for x in current_paras) + (2 if len(current_paras) > 1 else 0)
                else:
                    current_paras = [p]
                    current_len = len(p)
            else:
                # Single paragraph exceeds max_body_size: split via sliding window
                for start in range(0, p_len, step):
                    window = p[start : start + max_body_size].strip()
                    if window:
                        result.append(f"{prefix}{window}".strip())
                current_paras = []
                current_len = 0

    if current_paras:
        body = "\n\n".join(current_paras)
        result.append(f"{prefix}{body}".strip())

    return result


def _split_table_block(table_str: str, max_size: int) -> List[str]:
    """Splits large markdown tables across chunks while repeating table header rows."""
    lines = [line.strip() for line in table_str.strip().split("\n") if line.strip()]
    if len(lines) <= 2:
        return [table_str]

    if lines[0].startswith("|") and len(lines) > 1 and ("---" in lines[1] or ":-" in lines[1]):
        header_rows = lines[:2]
        data_rows = lines[2:]
    else:
        header_rows = [lines[0]]
        data_rows = lines[1:]

    header_block = "\n".join(header_rows) + "\n"
    chunks: List[str] = []
    curr: List[str] = []
    curr_len = len(header_block)

    for r in data_rows:
        if curr_len + len(r) + 1 <= max_size:
            curr.append(r)
            curr_len += len(r) + 1
        else:
            if curr:
                chunks.append(header_block + "\n".join(curr))
            curr = [r]
            curr_len = len(header_block) + len(r) + 1

    if curr:
        chunks.append(header_block + "\n".join(curr))

    return chunks


def _fallback_chunking(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback chunking when no headings exist."""
    subsections = split_subsections(text, "")
    if len(subsections) > 1:
        chunks = []
        for s in subsections:
            if len(s) <= chunk_size:
                chunks.append(s.strip())
            else:
                chunks.extend(_split_section_content(s, "", chunk_size, chunk_overlap))
        return chunks

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
    db_engine: Optional[Any] = None,
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

    docs = []
    for idx, chunk in enumerate(chunks):
        chunk_meta = {
            **meta,
            "doc_id": active_doc_id,
            "chunk_index": idx,
        }
        sec_match = re.match(r"^## Section:\s*([^\n]+)", chunk)
        if sec_match:
            chunk_meta["section"] = sec_match.group(1).strip()
        docs.append(Document(page_content=chunk, metadata=chunk_meta))
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
            db_engine=db_engine,
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
    db_engine: Optional[Any] = None,
) -> Optional[UploadedDocument]:
    """Records an uploaded document in the database registry."""
    from app.db import engine as default_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    eng = db_engine or default_engine
    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
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


async def get_uploaded_document(
    doc_id: str,
    db_engine: Optional[Any] = None,
) -> Optional[UploadedDocument]:
    """Retrieves an uploaded document record from SQLite."""
    from app.db import engine as default_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    eng = db_engine or default_engine
    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(UploadedDocument).where(UploadedDocument.doc_id == doc_id)
        result = await session.exec(statement)
        return result.first()


async def list_uploaded_documents(
    limit: int = 50,
    offset: int = 0,
    db_engine: Optional[Any] = None,
) -> List[UploadedDocument]:
    """Lists uploaded document records from SQLite ordered by creation time."""
    from app.db import engine as default_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    eng = db_engine or default_engine
    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = (
            select(UploadedDocument)
            .order_by(UploadedDocument.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        results = await session.exec(statement)
        return list(results.all())


async def delete_uploaded_document_record(
    doc_id: str,
    db_engine: Optional[Any] = None,
) -> Optional[UploadedDocument]:
    """Deletes an uploaded document record from SQLite."""
    from app.db import engine as default_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlmodel import select

    eng = db_engine or default_engine
    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        statement = select(UploadedDocument).where(UploadedDocument.doc_id == doc_id)
        result = await session.exec(statement)
        doc = result.first()
        if doc:
            await session.delete(doc)
            await session.commit()
            return doc
        return None
