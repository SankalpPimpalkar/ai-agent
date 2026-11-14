import unittest
import uuid
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from app.agent.builder import AgentRunner, build_agent
from app.agent.memory import (
    load_session_history,
    save_session_history,
    set_custom_checkpointer,
)
from app.agent.middleware import (
    prune_old_tool_outputs_middleware,
    summarize_context_middleware,
)
from app.config import Settings
from app.main import build_app
from app.tools.local import calculate_expression


class TestMemoryAndMiddlewares(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.memory_saver = MemorySaver()
        set_custom_checkpointer(self.memory_saver)

    async def asyncTearDown(self):
        set_custom_checkpointer(None)

    async def test_session_save_and_load(self):
        session_id = "test-session-xyz"
        msgs = [
            HumanMessage(content="Hello agent"),
            AIMessage(content="Hello human!"),
        ]
        await save_session_history(session_id, msgs, checkpointer=self.memory_saver)

        loaded = await load_session_history(session_id, checkpointer=self.memory_saver)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].content, "Hello agent")
        self.assertEqual(loaded[1].content, "Hello human!")

    async def test_sqlite_checkpointer_e2e(self):
        import os, tempfile
        from app.agent.memory import init_checkpointer
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_checkpointer.db")
        try:
            # Test direct SQLite checkpointer operations
            await init_checkpointer(db_path=db_path)
            session_id = "sqlite-sess-1"
            msgs = [
                HumanMessage(content="Stored in SQLite"),
                AIMessage(content="Acknowledged from SQLite"),
            ]
            await save_session_history(session_id, msgs, checkpointer=None, db_path=db_path)
            loaded = await load_session_history(session_id, checkpointer=None, db_path=db_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].content, "Stored in SQLite")
            self.assertEqual(loaded[1].content, "Acknowledged from SQLite")
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    async def test_multi_turn_agent_conversation_memory(self):
        fake_llm = FakeListChatModel(
            responses=[
                "Nice to meet you, Alice!",
                "Your name is Alice.",
            ]
        )
        runner = await build_agent(llm=fake_llm, checkpointer=self.memory_saver)
        session_id = f"user-{uuid.uuid4()}"

        # Turn 1
        res1 = await runner.execute("My name is Alice", session_id=session_id)
        self.assertEqual(res1, "Nice to meet you, Alice!")

        # Verify turn 1 was persisted
        history1 = await load_session_history(session_id, checkpointer=self.memory_saver)
        self.assertGreaterEqual(len(history1), 2)

        # Turn 2: asks what is my name
        res2 = await runner.execute("What is my name?", session_id=session_id)
        self.assertEqual(res2, "Your name is Alice.")

        # History should now have both turns
        history2 = await load_session_history(session_id, checkpointer=self.memory_saver)
        self.assertGreaterEqual(len(history2), 4)

    async def test_session_isolation(self):
        fake_llm = FakeListChatModel(responses=["Reply 1", "Reply 2"])
        runner = await build_agent(llm=fake_llm, checkpointer=self.memory_saver)

        sess_a = "session-a"
        sess_b = "session-b"

        await runner.execute("Message for A", session_id=sess_a)
        await runner.execute("Message for B", session_id=sess_b)

        history_a = await load_session_history(sess_a, checkpointer=self.memory_saver)
        history_b = await load_session_history(sess_b, checkpointer=self.memory_saver)

        self.assertIn("Message for A", [m.content for m in history_a])
        self.assertNotIn("Message for A", [m.content for m in history_b])

    def test_prune_old_tool_outputs_middleware(self):
        msgs = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Query 1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch_data", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(
                content="Huge raw output payload from database " * 30,
                tool_call_id="call_1",
                name="fetch_data",
            ),
            AIMessage(content="Processed query 1."),
            HumanMessage(content="Query 2"),
        ]

        pruned = prune_old_tool_outputs_middleware(msgs, min_length_to_prune=50)

        # The tool message should be pruned because an AIMessage follows it
        tool_msgs = [m for m in pruned if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("archived", tool_msgs[0].content)
        self.assertEqual(tool_msgs[0].tool_call_id, "call_1")
        self.assertEqual(tool_msgs[0].name, "fetch_data")
        # Ensure it is much shorter than original
        self.assertLess(len(tool_msgs[0].content), 100)

    async def test_summarize_context_middleware_overflow(self):
        msgs = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="Question one"),
            AIMessage(content="Answer one"),
            HumanMessage(content="Question two"),
            AIMessage(content="Answer two"),
            HumanMessage(content="Question three"),
            AIMessage(content="Answer three"),
            HumanMessage(content="Recent question"),
        ]

        # Trigger overflow with max_context_messages=5
        summarized = await summarize_context_middleware(
            msgs,
            max_context_chars=10000,
            max_context_messages=5,
            keep_recent=2,
        )

        self.assertLess(len(summarized), len(msgs))
        self.assertIsInstance(summarized[0], SystemMessage)
        self.assertEqual(summarized[0].content, "System prompt")
        # Second message should be the summary
        self.assertIn("Summary of previous conversation", summarized[1].content)
        # Recent message should be preserved
        self.assertEqual(summarized[-1].content, "Recent question")


class TestApiSessionManagement(unittest.TestCase):
    def setUp(self):
        self.memory_saver = MemorySaver()
        set_custom_checkpointer(self.memory_saver)
        self.settings = Settings(ENABLE_PERSISTENT_MEMORY=True)
        self.fake_llm = FakeListChatModel(responses=["Hello user!", "Welcome back!"])
        self.agent_runner = AgentRunner(llm=self.fake_llm, checkpointer=self.memory_saver)
        self.app = build_app(app_settings=self.settings, agent_runner=self.agent_runner)

    def tearDown(self):
        set_custom_checkpointer(None)

    def test_auto_generate_session_id(self):
        with TestClient(self.app) as client:
            res = client.post("/chat", json={"message": "Hi"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIsNotNone(data["session_id"])
            # Should be valid UUID format
            val = uuid.UUID(data["session_id"])
            self.assertEqual(str(val), data["session_id"])

    def test_reuse_existing_session_id(self):
        custom_session = "custom-session-1234"
        with TestClient(self.app) as client:
            res = client.post(
                "/chat",
                json={"message": "Hi", "session_id": custom_session},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["session_id"], custom_session)

    def test_stream_returns_session_id_header(self):
        with TestClient(self.app) as client:
            res = client.post(
                "/chat/stream",
                json={"message": "Stream hello", "session_id": "stream-sess-99"},
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers.get("X-Session-ID"), "stream-sess-99")

    def test_list_and_manage_conversations(self):
        with TestClient(self.app) as client:
            # Create session 1
            res1 = client.post(
                "/chat",
                json={"message": "How do I configure Docker containers?", "session_id": "sess-docker"},
            )
            self.assertEqual(res1.status_code, 200)

            # Create session 2
            res2 = client.post(
                "/chat",
                json={"message": "Tell me about Python asyncio concurrency.", "session_id": "sess-asyncio"},
            )
            self.assertEqual(res2.status_code, 200)

            # List conversations
            list_res = client.get("/conversations")
            self.assertEqual(list_res.status_code, 200)
            conversations = list_res.json()
            self.assertGreaterEqual(len(conversations), 2)

            session_ids = [c["session_id"] for c in conversations]
            self.assertIn("sess-docker", session_ids)
            self.assertIn("sess-asyncio", session_ids)

            # Verify summary and timestamp fields
            docker_item = next(c for c in conversations if c["session_id"] == "sess-docker")
            self.assertIn("Docker", docker_item["summary"])
            self.assertIn("last_message_at", docker_item)
            self.assertIn("created_at", docker_item)
            self.assertGreaterEqual(docker_item["message_count"], 1)

            # Get conversation details with message history
            detail_res = client.get("/conversations/sess-docker")
            self.assertEqual(detail_res.status_code, 200)
            detail = detail_res.json()
            self.assertEqual(detail["session_id"], "sess-docker")
            self.assertGreaterEqual(len(detail["messages"]), 1)

            # Delete conversation
            del_res = client.delete("/conversations/sess-docker")
            self.assertEqual(del_res.status_code, 204)

            # Confirm 404 after deletion
            get_after_del = client.get("/conversations/sess-docker")
            self.assertEqual(get_after_del.status_code, 404)


if __name__ == "__main__":
    unittest.main()
