import unittest
from pathlib import Path

from src.agent.tools.base import BaseTool
from src.agent.tools.calculator import CalculatorTool
from src.agent.tools.cli import CliCommandValidator, CliFileOperations, CliPathGuard, CliTool
from src.agent.tools.memory import MemoryFileStore, MemoryTool, SecretScanner
from src.agent.tools.planner import (
    PlannerTool,
    decompose_prompt,
    init_plan,
    mark_next_task_done,
    mark_plan_item_done,
)
from src.agent.tools.registry import ToolRegistry, create_default_registry
from src.agent.tools.builder import ToolAstValidator, ToolBuilderTool


class TestBaseToolAndRegistry(unittest.TestCase):
    def test_custom_tool_registration_and_dispatch(self):
        class EchoTool(BaseTool):
            @property
            def name(self) -> str:
                return "echo_tool"

            @property
            def description(self) -> str:
                return "Echoes the input back"

            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                }

            def run(self, msg: str = "", **kwargs) -> str:
                return f"echo: {msg}"

        tool = EchoTool()
        registry = ToolRegistry([tool])

        self.assertEqual(registry.dispatch("echo_tool", msg="hello"), "echo: hello")
        self.assertEqual(
            registry.dispatch("nonexistent"), "Error: unknown tool 'nonexistent'"
        )

        specs = registry.get_specs()
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].function.name, "echo_tool")

    def test_default_registry_contents(self):
        registry = create_default_registry()
        funcs = registry.get_functions()
        self.assertIn("calculator_tool", funcs)
        self.assertIn("cli_tool", funcs)
        self.assertIn("memory_tool", funcs)
        self.assertIn("planner_tool", funcs)
        self.assertIn("tool_builder", funcs)


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.calc = CalculatorTool()

    def test_binary_operations(self):
        self.assertEqual(self.calc.run(op1=2, operator="+", op2=3), "5")
        self.assertEqual(self.calc.run(op1=10, operator="-", op2=4), "6")
        self.assertEqual(self.calc.run(op1=3, operator="*", op2=5), "15")
        self.assertEqual(self.calc.run(op1=10, operator="/", op2=2), "5")

    def test_unary_operations(self):
        self.assertEqual(self.calc.run(op1=16, operator="sqrt"), "4")
        self.assertEqual(self.calc.run(op1=-5, operator="abs"), "5")

    def test_division_by_zero(self):
        res = self.calc.run(op1=5, operator="/", op2=0)
        self.assertEqual(res, "Error: division by zero")

    def test_unsupported_operator(self):
        res = self.calc.run(op1=5, operator="unknown_op", op2=2)
        self.assertTrue(res.startswith("Error: unsupported operator"))


class TestToolBuilderCollaborators(unittest.TestCase):
    def test_ast_validator_spec_generation(self):
        validator = ToolAstValidator()
        code = 'def add_nums(a: int, b: int = 0) -> int:\n    """Adds two integers."""\n    return a + b\n'
        props, required, doc = validator.build_spec(code, "add_nums")
        self.assertEqual(doc, "Adds two integers.")
        self.assertEqual(required, ["a"])
        self.assertIn("a", props)
        self.assertEqual(props["a"]["type"], "integer")
        self.assertIn("b", props)

    def test_ast_validator_missing_named_function(self):
        validator = ToolAstValidator()
        code = "def other_func():\n    pass\n"
        with self.assertRaises(ValueError) as ctx:
            validator.build_spec(code, "target_func")
        self.assertIn("code must define a function named exactly 'target_func'", str(ctx.exception))


class TestCliCollaborators(unittest.TestCase):
    def test_dev_server_rejection(self):
        guard = CliPathGuard()
        validator = CliCommandValidator(guard)
        with self.assertRaises(ValueError) as ctx:
            validator.execute(project="test_proj", path=".", command="npm run dev")
        self.assertIn("Dev servers never exit", str(ctx.exception))

    def test_empty_command_rejection(self):
        guard = CliPathGuard()
        validator = CliCommandValidator(guard)
        with self.assertRaises(ValueError) as ctx:
            validator.execute(project="test_proj", path=".", command="   ")
        self.assertIn("Empty command", str(ctx.exception))


class TestPlannerTool(unittest.TestCase):
    def test_decompose_prompt_regex(self):
        prompt = "Build a regular expression matcher from scratch without using any built-in regex library. Store the matching strings under 'matches'."
        tasks, tests = decompose_prompt(prompt, "regex-engine")
        self.assertGreaterEqual(len(tasks), 3)
        self.assertGreaterEqual(len(tests), 2)
        self.assertTrue(any("matches" in t for t in tasks))

    def test_decompose_prompt_interpreter(self):
        prompt = "Design and build an interpreter for a small programming language of your own. Store the outputs under 'interp_results'."
        tasks, tests = decompose_prompt(prompt, "mini-interpreter")
        self.assertGreaterEqual(len(tasks), 3)
        self.assertGreaterEqual(len(tests), 2)
        self.assertTrue(any("interp_results" in t for t in tasks + tests))

    def test_init_plan_and_mark_done(self):
        prompt = "Build a todo cli application"
        content, plan_path = init_plan("todo-cli", prompt=prompt)
        self.assertTrue(plan_path.exists())
        self.assertIn("## Tasks", content)
        self.assertIn("## Tests", content)
        # Should have multiple tasks and tests
        task_count = content.count("- [ ]")
        self.assertGreaterEqual(task_count, 4)

        # Mark next task done
        ok, line = mark_next_task_done("todo-cli", section="tasks")
        self.assertTrue(ok)
        updated_content = plan_path.read_text(encoding="utf-8")
        self.assertIn("- [x]", updated_content)
        self.assertIn("- [ ]", updated_content)

        # Mark next test done
        ok2, line2 = mark_next_task_done("todo-cli", section="tests")
        self.assertTrue(ok2)

        # Use tool to mark done
        tool = PlannerTool()
        res = tool.run(action="mark_done", project="todo-cli", task="1")
        self.assertIn("Marked task done", res)


if __name__ == "__main__":
    unittest.main()
