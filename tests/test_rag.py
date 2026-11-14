import unittest
from langchain_core.vectorstores import InMemoryVectorStore
from app.rag.ingest import chunk_text, ingest_document
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
