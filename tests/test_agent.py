import unittest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.agent.builder import AgentRunner, build_agent
from app.schemas.mcp import MCPConfigFile, MCPServerConfig
from app.tools.local import calculate_expression, get_current_time
from app.tools.mcp import get_mcp_connections, load_mcp_config_file
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

    async def test_tools_registry_flags(self):
        # When all flags/sources are disabled
        tools = await get_all_tools(include_local=False, include_rag=False, include_mcp=False)
        self.assertEqual(len(tools), 0)

    def test_mcp_config_file_and_schema_validation(self):
        # 1. Test parsing real mcp_servers.json
        cfg = load_mcp_config_file()
        self.assertIsNotNone(cfg)
        self.assertIn("git", cfg.mcpServers)

        conns = get_mcp_connections(cfg)
        self.assertIn("git", conns)
        self.assertEqual(conns["git"]["transport"], "stdio")
        self.assertIn("mcp_server_git", " ".join(conns["git"]["args"]))

        # 2. Test schema validation on stdio and sse configurations
        stdio_srv = MCPServerConfig(command="python", args=["-m", "demo"])
        self.assertEqual(stdio_srv.transport, "stdio")

        sse_srv = MCPServerConfig(url="http://localhost:8000/sse")
        self.assertEqual(sse_srv.transport, "sse")

        # 3. Test validation failure when neither command nor url provided
        with self.assertRaises(ValueError):
            MCPServerConfig(env={"VAR": "1"})

        # 4. Test disabled flag
        disabled_cfg = MCPConfigFile.from_raw_data({
            "mcpServers": {
                "active_git": {"command": "python", "args": ["-m", "mcp_server_git"]},
                "disabled_srv": {"command": "echo", "disabled": True},
            }
        })
        active_conns = get_mcp_connections(disabled_cfg)
        self.assertIn("active_git", active_conns)
        self.assertNotIn("disabled_srv", active_conns)


if __name__ == "__main__":
    unittest.main()
