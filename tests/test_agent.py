import unittest
from dataclasses import dataclass
from typing import List, Optional
from ollama import Message

from src.agent.core import Agent, LoopBreakers
from src.agent.conversation import ConversationHistory, PromptBuilder
from src.agent.tools.memory import MemoryService
from src.agent.tools.registry import ToolRegistry


@dataclass
class FakeMessage:
    role: str
    content: str
    tool_calls: Optional[list] = None
    thinking: Optional[str] = None


@dataclass
class FakeResponse:
    message: FakeMessage
    prompt_eval_count: int = 10
    eval_count: int = 20


class FakeChatClient:
    def __init__(self, responses: Optional[List[FakeResponse]] = None):
        self.responses = list(responses) if responses else []
        self.call_history: List[dict] = []

    def chat(self, model: str, messages: list, tools=None, think=False, options=None):
        self.call_history.append(
            {"model": model, "messages": messages, "tools": tools}
        )
        if self.responses:
            return self.responses.pop(0)
        return FakeResponse(
            message=FakeMessage(role="assistant", content="Default test response")
        )


class TestLoopBreakers(unittest.TestCase):
    def setUp(self):
        self.breakers = LoopBreakers(max_consecutive_errors=4, max_repeated_calls=3)

    def test_consecutive_errors_trigger(self):
        for step in range(1, 4):
            res = self.breakers.check("dummy", {}, "Error: bad", step)
            self.assertNotIn("Too many errors", res)
            self.assertNotIn("errors", self.breakers.triggered_breakers)

        # 4th error triggers breaker
        res = self.breakers.check("dummy", {}, "Error: bad", 4)
        self.assertIn("Too many errors in a row", res)
        self.assertIn("errors", self.breakers.triggered_breakers)

    def test_consecutive_errors_reset_on_success(self):
        self.breakers.check("dummy", {}, "Error: bad", 1)
        self.breakers.check("dummy", {}, "Error: bad", 2)
        # Success resets counter
        self.breakers.check("dummy", {}, "Success!", 3)
        self.assertEqual(self.breakers.consecutive_errors, 0)

        res = self.breakers.check("dummy", {}, "Error: bad", 4)
        self.assertNotIn("Too many errors", res)

    def test_repeat_calls_trigger(self):
        self.breakers.check("calc", {"a": 1}, "42", 1)
        self.breakers.check("calc", {"a": 1}, "42", 2)
        res = self.breakers.check("calc", {"a": 1}, "42", 3)
        self.assertIn("repeated this exact call 3 times", res)
        self.assertIn("repeat", self.breakers.triggered_breakers)


class TestConversationHistory(unittest.TestCase):
    def test_append_and_access(self):
        hist = ConversationHistory()
        hist.append(Message(role="user", content="hello"))
        hist.append(Message(role="assistant", content="hi"))
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0].content, "hello")
        self.assertEqual(hist[1].content, "hi")

    def test_extend_and_clear(self):
        hist = ConversationHistory()
        hist.extend([Message(role="user", content="msg1"), Message(role="user", content="msg2")])
        self.assertEqual(len(hist), 2)
        hist.clear()
        self.assertEqual(len(hist), 0)


class TestPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()

    def test_tool_use_mode_includes_memory_index(self):
        msgs = self.builder.build_messages(
            chat_history=[],
            tool_use=True,
            system_prompt="Custom System Prompt",
            memory_index_text="file1.txt: note",
        )
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Custom System Prompt", msgs[0]["content"])
        self.assertIn("file1.txt: note", msgs[0]["content"])

    def test_no_tool_mode(self):
        msgs = self.builder.build_messages(
            chat_history=[],
            tool_use=False,
            system_prompt="Simple Bot",
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("You have no tools in this mode", msgs[0]["content"])


class TestAgentWithFakeClient(unittest.TestCase):
    def test_agent_run_terminates_on_text_response(self):
        client = FakeChatClient([
            FakeResponse(message=FakeMessage(role="assistant", content="The capital is Paris."))
        ])
        registry = ToolRegistry([])
        agent = Agent(client=client, tool_registry=registry)

        output = agent.run(user_input="What is the capital of France?", tool_use=False)
        self.assertEqual(output.get("answer"), "The capital is Paris.")
        self.assertEqual(output.get("steps"), 1)
        self.assertEqual(len(client.call_history), 1)


if __name__ == "__main__":
    unittest.main()
