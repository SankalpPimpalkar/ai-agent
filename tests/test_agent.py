import unittest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agent.builder import AgentRunner, build_agent
from app.tools.local import calculate_expression, get_current_time
from app.tools.registry import get_all_tools


class TestAgentAndTools(unittest.IsolatedAsyncioTestCase):
    async def test_build_messages(self):
        runner = AgentRunner()
        history = [
            HumanMessage(content="User message"),
            AIMessage(content="Bot reply"),
        ]
        msgs = runner.build_messages(
            message="Next query",
            history=history,
            system_prompt="Custom system",
        )
        self.assertEqual(len(msgs), 4)
        self.assertIsInstance(msgs[0], SystemMessage)
        self.assertEqual(msgs[0].content, "Custom system")
        self.assertEqual(msgs[1].content, "User message")
        self.assertEqual(msgs[2].content, "Bot reply")
        self.assertIsInstance(msgs[3], HumanMessage)
        self.assertEqual(msgs[3].content, "Next query")

    async def test_agent_execution_sync_and_stream(self):
        fake_llm = FakeListChatModel(responses=["Agent response"])
        runner = await build_agent(llm=fake_llm, tools=[])

        answer = await runner.execute("Test prompt")
        self.assertEqual(answer, "Agent response")

        stream_llm = FakeListChatModel(responses=["Stream chunk"])
        stream_runner = await build_agent(llm=stream_llm, tools=[])
        chunks = []
        async for chunk in stream_runner.execute_stream("Test stream"):
            chunks.append(chunk)
        self.assertEqual("".join(chunks), "Stream chunk")

    def test_local_tools(self):
        time_result = get_current_time.invoke({})
        self.assertTrue(isinstance(time_result, str))
        self.assertIn("T", time_result)

        calc_result = calculate_expression.invoke({"expression": "2 + 3 * 4"})
        self.assertEqual(calc_result, "14")

        calc_err = calculate_expression.invoke({"expression": "1 / 0"})
        self.assertIn("division by zero", calc_err)

    async def test_tools_registry(self):
        # By default flags are false, returns empty list
        tools = await get_all_tools()
        self.assertEqual(len(tools), 0)


if __name__ == "__main__":
    unittest.main()
