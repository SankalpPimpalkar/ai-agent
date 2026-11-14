import unittest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from app.agent.builder import AgentRunner
from app.config import Settings
from app.main import build_app
from app.rag.store import InMemoryVectorStore, reset_vector_store


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)

    async def aembed_query(self, text):
        return self.embed_query(text)


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        reset_vector_store()
        self.settings = Settings(ENABLE_RAG=False)
        self.fake_llm = FakeListChatModel(responses=["Hello from LangChain!"])
        self.agent_runner = AgentRunner(llm=self.fake_llm, tools=[])
        self.app = build_app(app_settings=self.settings, agent_runner=self.agent_runner)

    def test_health_endpoint(self):
        with TestClient(self.app) as client:
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["app_name"], self.settings.APP_NAME)
            self.assertFalse(data["rag_enabled"])

    def test_chat_endpoint_sync(self):
        with TestClient(self.app) as client:
            payload = {"message": "Hello AI"}
            res = client.post("/chat", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["response"], "Hello from LangChain!")
            self.assertEqual(data["model"], self.settings.MODEL)

    def test_chat_endpoint_empty_message(self):
        with TestClient(self.app) as client:
            res = client.post("/chat", json={"message": "   "})
            self.assertEqual(res.status_code, 422)

    def test_chat_stream_endpoint(self):
        with TestClient(self.app) as client:
            payload = {"message": "Hello Stream"}
            res = client.post("/chat/stream", json=payload)
            self.assertEqual(res.status_code, 200)
            self.assertIn("text/event-stream", res.headers.get("content-type", ""))
            self.assertIn("Hello from LangChain!", res.text)

    def test_chat_stream_endpoint_with_tool_calling(self):
        from app.tools.local import calculate_expression
        from tests.test_agent import MockStreamingToolLLM

        mock_llm = MockStreamingToolLLM()
        agent_runner = AgentRunner(llm=mock_llm, tools=[calculate_expression])
        app = build_app(app_settings=self.settings, agent_runner=agent_runner)

        with TestClient(app) as client:
            payload = {"message": "Compute 10 + 20", "stream_tools": True}
            res = client.post("/chat/stream", json=payload)
            self.assertEqual(res.status_code, 200)
            self.assertIn("text/event-stream", res.headers.get("content-type", ""))
            self.assertIn("[Calling tool 'calculate_expression'...]", res.text)
            self.assertIn("The result is 30.", res.text)

    def test_documents_endpoint_disabled(self):
        with TestClient(self.app) as client:
            payload = {"content": "Sample document content"}
            res = client.post("/documents", json=payload)
            self.assertEqual(res.status_code, 403)
            self.assertIn("RAG feature is disabled", res.json()["detail"])

    def test_documents_endpoint_enabled(self):
        rag_settings = Settings(ENABLE_RAG=True)
        rag_app = build_app(app_settings=rag_settings, agent_runner=self.agent_runner)
        # Mock vector store with fake embeddings
        fake_store = InMemoryVectorStore(FakeEmbeddings())
        rag_app.dependency_overrides = {}

        from app.api.deps import get_vector_store_dep
        rag_app.dependency_overrides[get_vector_store_dep] = lambda: fake_store

        with TestClient(rag_app) as client:
            payload = {"content": "First paragraph.\n\nSecond paragraph."}
            res = client.post("/documents", json=payload)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertGreaterEqual(data["chunks_count"], 1)

    def test_documents_endpoint_file_upload(self):
        rag_settings = Settings(ENABLE_RAG=True)
        rag_app = build_app(app_settings=rag_settings, agent_runner=self.agent_runner)
        fake_store = InMemoryVectorStore(FakeEmbeddings())
        from app.api.deps import get_vector_store_dep
        rag_app.dependency_overrides[get_vector_store_dep] = lambda: fake_store

        with TestClient(rag_app) as client:
            files = {"file": ("manual.txt", b"This is the user manual content.", "text/plain")}
            res = client.post("/documents", files=files)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertGreaterEqual(data["chunks_count"], 1)
            self.assertIn("manual.txt", data["message"])

    def test_documents_endpoint_empty_payload(self):
        rag_settings = Settings(ENABLE_RAG=True)
        rag_app = build_app(app_settings=rag_settings, agent_runner=self.agent_runner)
        fake_store = InMemoryVectorStore(FakeEmbeddings())
        from app.api.deps import get_vector_store_dep
        rag_app.dependency_overrides[get_vector_store_dep] = lambda: fake_store

        with TestClient(rag_app) as client:
            res = client.post("/documents", json={"content": "  "})
            self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
