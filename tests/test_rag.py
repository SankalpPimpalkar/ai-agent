import io
import tempfile
import unittest
import zipfile
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from app.rag.ingest import (
    chunk_text,
    delete_document_from_store,
    extract_text_from_file,
    ingest_document,
    linearize_markdown_table,
    split_subsections,
)
from app.rag.store import get_vector_store, reset_vector_store


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


class TestRagPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_vector_store()
        self.embeddings = FakeEmbeddings()
        self.store = InMemoryVectorStore(self.embeddings)

    def test_chunk_text(self):
        sample = "A" * 2500
        chunks = chunk_text(sample, chunk_size=1000, chunk_overlap=200)
        self.assertGreater(len(chunks), 1)

    def test_extract_text_plain_and_markdown(self):
        txt_bytes = b"Hello from plain text file."
        self.assertEqual(extract_text_from_file("notes.txt", txt_bytes), "Hello from plain text file.")

        md_bytes = b"# Header\n- item 1\n- item 2"
        self.assertIn("Header", extract_text_from_file("README.md", md_bytes))

    def test_extract_text_docx(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            xml_data = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>Docx paragraph one.</w:t></w:r></w:p>
                <w:p><w:r><w:t>Docx paragraph two.</w:t></w:r></w:p>
              </w:body>
            </w:document>"""
            zf.writestr("word/document.xml", xml_data)
        extracted = extract_text_from_file("sample.docx", buf.getvalue())
        self.assertIn("Docx paragraph one.", extracted)
        self.assertIn("Docx paragraph two.", extracted)

    def test_extract_text_pdf(self):
        import pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        # Should not raise exception
        result = extract_text_from_file("blank.pdf", buf.getvalue())
        self.assertIsInstance(result, str)

    async def test_ingest_and_retrieve(self):
        content = "LangChain provides abstractions for LLMs, prompt templates, and vector stores."
        doc_id, count = await ingest_document(content, vector_store=self.store)
        self.assertGreaterEqual(count, 1)
        self.assertTrue(bool(doc_id))

        retriever = self.store.as_retriever(search_kwargs={"k": 1})
        docs = await retriever.ainvoke("What does LangChain provide?")
        self.assertGreater(len(docs), 0)
        self.assertIn("LangChain", docs[0].page_content)

    async def test_delete_document_from_store(self):
        content = "Confidential document to be deleted."
        doc_id, count = await ingest_document(content, doc_id="doc-to-del", vector_store=self.store)
        self.assertEqual(doc_id, "doc-to-del")

        retriever = self.store.as_retriever(search_kwargs={"k": 2})
        docs_before = await retriever.ainvoke("Confidential")
        self.assertGreater(len(docs_before), 0)

        # Delete document entirely
        await delete_document_from_store(doc_id=doc_id, chunks_count=count, vector_store=self.store)

        docs_after = await retriever.ainvoke("Confidential")
        self.assertEqual(len(docs_after), 0)

    def test_chunk_text_section_headings(self):
        sample_md = """# Privacy Policy
Introductory text about the company.

## 1. Information We Collect
We collect personal information when you create an account.

### 1.1 Account Data
We collect name, email address, and authentication credentials.

## 2. Data Retention
We retain data according to regulatory requirements:

| Data Type | Retention Period |
| Support chats | 3 years |
| Call recordings | 12 months |
"""
        chunks = chunk_text(sample_md, chunk_size=500, chunk_overlap=100)
        self.assertGreaterEqual(len(chunks), 4)

        # Verify heading hierarchy is prepended
        self.assertTrue(chunks[0].startswith("## Section: Privacy Policy"))
        self.assertIn("Introductory text", chunks[0])

        self.assertTrue(chunks[1].startswith("## Section: Privacy Policy > 1. Information We Collect"))
        self.assertTrue(chunks[2].startswith("## Section: Privacy Policy > 1. Information We Collect > 1.1 Account Data"))

        # Verify table chunk preserves section heading and linearized table contents
        retention_chunk = next(c for c in chunks if "Support chats" in c)
        self.assertTrue(retention_chunk.startswith("## Section: Privacy Policy > 2. Data Retention"))
        self.assertIn("Support chats: retained 3 years", retention_chunk)
        self.assertIn("Call recordings: retained 12 months", retention_chunk)

    def test_split_large_table_repeats_header(self):
        sample_md = """# Data Dictionary
## 1. Retention Table

| Category | Retention Duration | Regulatory Basis |
|---|---|---|
| Customer Chat Logs | 3 Years | Consumer Protection Act |
| Audio Call Recordings | 12 Months | Telecommunications Regulation |
| KYC Identity Documents | 5 Years After Account Closure | Anti Money Laundering |
| Transaction History | 7 Years | Financial Compliance Standard |
"""
        # Very small chunk size to force table splitting across chunks
        chunks = chunk_text(sample_md, chunk_size=160, chunk_overlap=30)
        table_chunks = [c for c in chunks if "## Section: Data Dictionary > 1. Retention Table" in c]
        self.assertGreater(len(table_chunks), 1)

        # Every table sub-chunk must retain linearized table data
        combined = " ".join(table_chunks)
        self.assertIn("Customer Chat Logs: retained 3 Years", combined)

    def test_extract_text_pdf_pymupdf4llm(self):
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "# ShopNest Security Policy\n\nAll passwords must be salted and hashed.")
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        extracted = extract_text_from_file("security.pdf", buf.getvalue())
        self.assertIn("ShopNest Security Policy", extracted)
        self.assertIn("passwords must be salted", extracted)

    async def test_ingest_document_records_section_metadata(self):
        sample = """# API Guide
## Endpoints
GET /users returns user list.
"""
        doc_id, count = await ingest_document(sample, vector_store=self.store)
        self.assertGreaterEqual(count, 1)

        # Check that stored document in vector store has section in metadata
        retriever = self.store.as_retriever(search_kwargs={"k": 2})
        docs = await retriever.ainvoke("endpoints")
        self.assertGreater(len(docs), 0)
        self.assertIn("section", docs[0].metadata)
        self.assertIn("API Guide", docs[0].metadata["section"])

    async def test_chroma_persistent_store(self):
        temp_dir = tempfile.mkdtemp()
        chroma_store = Chroma(
            collection_name="test_persistent",
            embedding_function=self.embeddings,
            persist_directory=temp_dir,
        )
        doc_id, count = await ingest_document(
            "Persistent information in ChromaDB.",
            doc_id="chroma-doc-1",
            vector_store=chroma_store,
        )
        self.assertEqual(doc_id, "chroma-doc-1")

        search_res = await chroma_store.asimilarity_search("Persistent", k=1)
        self.assertGreater(len(search_res), 0)
        self.assertIn("Persistent information", search_res[0].page_content)

        # Delete cleanly using Chroma
        await delete_document_from_store(doc_id="chroma-doc-1", vector_store=chroma_store)
        search_after = await chroma_store.asimilarity_search("Persistent", k=1)
        self.assertEqual(len(search_after), 0)

    def test_split_subsections_numbered_and_unnumbered(self):
        text = (
            "**12.1 ShopNest Prime.** Viewing history linked. "
            "**12.2 ShopNest Voice.** Cloud recordings. "
            "**12.3 ShopNest Pay.** Anti-money-laundering 5 years. "
            "**Nevada.** Nevada residents opt out. "
            "**EEA/UK.** Supervisory complaint."
        )
        chunks = split_subsections(text, "## Section: Policy > Specific Programs")
        self.assertEqual(len(chunks), 5)
        self.assertIn("12.1 ShopNest Prime.", chunks[0])
        self.assertIn("12.2 ShopNest Voice.", chunks[1])
        self.assertIn("12.3 ShopNest Pay.", chunks[2])
        self.assertIn("Nevada.", chunks[3])
        self.assertIn("EEA/UK.", chunks[4])
        for c in chunks:
            self.assertTrue(c.startswith("## Section: Policy > Specific Programs"))

    def test_split_subsections_preserves_colon_bold_phrase(self):
        text = (
            "General security practices with TLS 1.3 and AES-256. "
            "**Breach notification:** we notify affected users without undue delay, and within 72 hours."
        )
        chunks = split_subsections(text, "## Section: Policy > Security")
        self.assertEqual(len(chunks), 1)
        self.assertIn("**Breach notification:**", chunks[0])
        self.assertIn("within 72 hours", chunks[0])

    def test_chunk_text_section_12_and_13_subsections(self):
        sample = """# ShopNest Privacy Policy
## 12. Specific Programs
**12.1 Prime.** Prime membership details. **12.2 Voice.** Voice recording details. **12.3 Pay.** AML rules require keeping records for 5 years.

## 13. Regional Disclosures
**California (CCPA).** CCPA disclosures. **Nevada.** Nevada opt out. **EEA/UK.** Lodge complaint with Irish DPC.
"""
        chunks = chunk_text(sample, chunk_size=1000)
        # 12.1, 12.2, 12.3, California, Nevada, EEA/UK should all be distinct chunks
        pay_chunk = next((c for c in chunks if "12.3 Pay." in c), None)
        self.assertIsNotNone(pay_chunk)
        self.assertTrue(pay_chunk.startswith("## Section: ShopNest Privacy Policy > 12. Specific Programs"))
        self.assertNotIn("12.1 Prime.", pay_chunk)

        eea_chunk = next((c for c in chunks if "EEA/UK." in c), None)
        self.assertIsNotNone(eea_chunk)
        self.assertTrue(eea_chunk.startswith("## Section: ShopNest Privacy Policy > 13. Regional Disclosures"))
        self.assertIn("Irish DPC", eea_chunk)

    def test_linearize_markdown_table_pattern(self):
        table_sec6 = (
            "|Data type|Retention period|Notes|\n"
            "|---|---|---|\n"
            "|Customer support chats|3 years|Call recordings: 12 months|\n"
            "|Order history and invoices|7 years|Tax and accounting law|\n"
        )
        res = linearize_markdown_table(table_sec6)
        self.assertIn("Customer support chats: retained 3 years. Call recordings: 12 months", res)
        self.assertIn("Order history and invoices: retained 7 years. Tax and accounting law", res)

        table_appendix = (
            "|Question|Answer|\n"
            "|---|---|\n"
            "|Minimum account age|18|\n\n\n\n"
            "|Question|Answer|\n"
            "|---|---|\n"
            "|Support call recordings retained|12 months|\n"
        )
        res_app = linearize_markdown_table(table_appendix)
        self.assertIn("Minimum account age: 18", res_app)
        self.assertIn("Support call recordings retained: 12 months", res_app)
        # Verify duplicate header was removed
        self.assertNotIn("Question: Answer", res_app)


if __name__ == "__main__":
    unittest.main()
