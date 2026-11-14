import io
import unittest
import zipfile
from langchain_core.vectorstores import InMemoryVectorStore
from app.rag.ingest import chunk_text, extract_text_from_file, ingest_document
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
        count = await ingest_document(content, vector_store=self.store)
        self.assertGreaterEqual(count, 1)

        retriever = self.store.as_retriever(search_kwargs={"k": 1})
        docs = await retriever.ainvoke("What does LangChain provide?")
        self.assertGreater(len(docs), 0)
        self.assertIn("LangChain", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
