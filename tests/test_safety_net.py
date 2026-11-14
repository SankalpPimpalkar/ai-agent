import os
import shutil
import unittest
from pathlib import Path
from ollama import Message

from src.agent.core.types import AgentState
from src.agent.tools.cli import _safe_path, safe_resolve_path
from src.agent.tools.memory import MEMORY_DIR, memory_tool
from src.agent.conversation.history import trim_chat


class TestTrimChat(unittest.TestCase):
    def test_trim_chat_preserves_system_message(self):
        history = [Message(role="system", content="You are a helper.")]
        for i in range(15):
            history.append(Message(role="user", content=f"msg {i}"))
        trim_chat(history, max_messages=10)

        # First message must still be system message
        self.assertEqual(history[0].role, "system")
        self.assertEqual(history[0].content, "You are a helper.")
        # Remaining messages should be trimmed to at most max_messages
        self.assertLessEqual(len(history), 11)

    def test_trim_chat_handles_short_history(self):
        history = [
            Message(role="system", content="Sys"),
            Message(role="user", content="Hello"),
        ]
        trim_chat(history, max_messages=10)
        self.assertEqual(len(history), 2)


class TestMemoryToolActions(unittest.TestCase):
    def setUp(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.test_file = "test_safety_profile"
        for ext in (".md", ".txt"):
            p = MEMORY_DIR / f"{self.test_file}{ext}"
            if p.exists():
                p.unlink()

    def tearDown(self):
        for ext in (".md", ".txt"):
            p = MEMORY_DIR / f"{self.test_file}{ext}"
            if p.exists():
                p.unlink()

    def test_create_and_read_memory(self):
        res = memory_tool(
            action="create",
            file=self.test_file,
            content="User Profile\nname: Alice\nrole: Engineer",
        )
        self.assertIn("Created", res)

        read_res = memory_tool(action="read", file=self.test_file)
        self.assertIn("Alice", read_res)
        self.assertIn("Engineer", read_res)

    def test_append_memory_prevents_duplicate(self):
        memory_tool(
            action="create",
            file=self.test_file,
            content="User Profile\nrole: Engineer",
        )
        # Duplicate append
        dup_res = memory_tool(
            action="append",
            file=self.test_file,
            content="role: Engineer",
        )
        self.assertIn("already exists", dup_res)

    def test_update_and_delete_lines(self):
        memory_tool(
            action="create",
            file=self.test_file,
            content="Line 1\nLine 2\nLine 3",
        )
        update_res = memory_tool(
            action="update_lines",
            file=self.test_file,
            content="Updated Line 2",
            start_line=2,
            end_line=2,
        )
        self.assertIn("Updated lines", update_res)

        read_res = memory_tool(action="read", file=self.test_file)
        self.assertIn("Updated Line 2", read_res)

        delete_res = memory_tool(
            action="delete_lines",
            file=self.test_file,
            start_line=2,
            end_line=2,
        )
        self.assertIn("Deleted lines", delete_res)

    def test_memory_rejects_secrets(self):
        res = memory_tool(
            action="create",
            file=self.test_file,
            content="api_key: sk-1234567890abcdef1234",
        )
        self.assertIn("Error: do not store secrets", res)


class TestCliPathSafety(unittest.TestCase):
    def test_safe_resolve_path_within_bounds(self):
        base = Path("/tmp/test_sandbox").resolve()
        base.mkdir(parents=True, exist_ok=True)
        resolved = safe_resolve_path(base, "sub/file.txt")
        self.assertTrue(str(resolved).startswith(str(base)))

    def test_safe_resolve_path_escapes_raises(self):
        base = Path("/tmp/test_sandbox").resolve()
        with self.assertRaises(ValueError):
            safe_resolve_path(base, "../../etc/passwd")

    def test_safe_path_escapes_project_raises(self):
        with self.assertRaises(ValueError):
            _safe_path("../../secret.txt", "my-project")

    def test_normalize_strips_nested_artifacts_prefixes(self):
        from src.agent.tools.cli.path_guard import CliPathGuard
        guard = CliPathGuard()

        proj, path = guard.normalize("todo-cli", "artifacts/todo-cli/todo.py", {})
        self.assertEqual(proj, "todo-cli")
        self.assertEqual(path, "todo.py")

        proj, path = guard.normalize("artifacts", "todo-cli/todo.py", {})
        self.assertEqual(proj, "todo-cli")
        self.assertEqual(path, "todo.py")

        proj, path = guard.normalize("todo-cli", "artifacts/todo-cli", {})
        self.assertEqual(proj, "todo-cli")
        self.assertEqual(path, ".")

        resolved = guard.safe_path("artifacts/todo-cli/todo.py", "todo-cli")
        self.assertFalse(str(resolved).endswith("/artifacts/todo-cli/artifacts/todo-cli/todo.py"))
        self.assertTrue(str(resolved).endswith("/todo-cli/todo.py"))


class TestLoopBreakers(unittest.TestCase):
    def test_consecutive_errors_breaker(self):
        state = AgentState()
        for i in range(3):
            res = state.check_breakers("calculator_tool", {"op1": 1}, "Error: fail", i + 1)
            self.assertNotIn("Too many errors", res)
            self.assertNotIn("errors", state.breakers)

        # 4th consecutive error triggers breaker
        res = state.check_breakers("calculator_tool", {"op1": 1}, "Error: fail", 4)
        self.assertIn("Too many errors in a row", res)
        self.assertIn("errors", state.breakers)

    def test_consecutive_errors_resets_on_success(self):
        state = AgentState()
        state.check_breakers("calculator_tool", {}, "Error: fail", 1)
        state.check_breakers("calculator_tool", {}, "Error: fail", 2)
        # Success resets error count
        state.check_breakers("calculator_tool", {}, "Success", 3)
        self.assertEqual(state.errors, 0)

        # Next error starts at 1, does not trigger breaker
        res = state.check_breakers("calculator_tool", {}, "Error: fail", 4)
        self.assertNotIn("errors", state.breakers)

    def test_repeat_call_breaker(self):
        state = AgentState()
        args = {"command": "ls"}
        for i in range(2):
            res = state.check_breakers("cli_tool", args, "output.txt", i + 1)
            self.assertNotIn("repeated this exact call 3 times", res)
            self.assertNotIn("repeat", state.breakers)

        # 3rd identical call with same result triggers breaker
        res = state.check_breakers("cli_tool", args, "output.txt", 3)
        self.assertIn("repeated this exact call 3 times with the same result", res)
        self.assertIn("repeat", state.breakers)


class TestConfigAndDependencies(unittest.TestCase):
    def test_typed_settings_groups(self):
        from src.agent.settings import SETTINGS
        # Check typed groups
        self.assertTrue(hasattr(SETTINGS, "db"))
        self.assertTrue(hasattr(SETTINGS, "model"))
        self.assertTrue(hasattr(SETTINGS, "paths"))
        self.assertTrue(hasattr(SETTINGS, "prompts"))

        self.assertIn("sqlite", SETTINGS.db.url)
        self.assertEqual(SETTINGS.model.default_model, SETTINGS.DEFAULT_MODEL)
        self.assertEqual(SETTINGS.paths.artifacts_folder, "artifacts")
        self.assertTrue(isinstance(SETTINGS.prompts.system_prompt, str))

    def test_build_app_composition_root(self):
        from src.agent.main import build_app
        from src.agent.settings import SETTINGS
        custom_app = build_app(settings=SETTINGS)
        self.assertEqual(custom_app.state.settings, SETTINGS)
        routes = [r.path for r in custom_app.routes if hasattr(r, "path")]
        self.assertIn("/api/chat", routes)
        self.assertIn("/api/tool", routes)
        self.assertIn("/api/stream", routes)


if __name__ == "__main__":
    unittest.main()
