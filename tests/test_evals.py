import tempfile
import unittest
from pathlib import Path

from evals.layers import (
    BaseEvalLayer,
    OutputEvalLayer,
    ProcessEvalLayer,
    ToolEvalLayer,
)
from evals.models import Case, EvalRunRecord, Run
from evals.storage import EvalStorage
from src.agent.core.types import Trace, TraceArgs


class TestEvalLayers(unittest.TestCase):
    def test_output_eval_layer(self):
        layer = OutputEvalLayer()
        case = Case(name="test", prompt="test", expected_output=["result", "42"])

        # Success case
        run_ok = Run(steps=1, tokens=10, answer="The final result is 42.")
        self.assertEqual(layer.check(case, run_ok), [])

        # Missing output
        run_missing = Run(steps=1, tokens=10, answer="The final value is 100.")
        fails = layer.check(case, run_missing)
        self.assertEqual(len(fails), 2)
        self.assertIn("Expected output missing: 'result'", fails)
        self.assertIn("Expected output missing: '42'", fails)

    def test_process_eval_layer(self):
        layer = ProcessEvalLayer()
        case = Case(name="test", prompt="test", max_steps=5)

        run_ok = Run(steps=3, tokens=10, answer="done")
        self.assertEqual(layer.check(case, run_ok), [])

        run_exceeded = Run(steps=6, tokens=10, answer="done")
        fails = layer.check(case, run_exceeded)
        self.assertEqual(len(fails), 1)
        self.assertIn("Too many steps: 6 (max: 5)", fails[0])

    def test_tool_eval_layer(self):
        layer = ToolEvalLayer()
        case = Case(name="test", prompt="test", forbid_tools=["calculator_tool"])

        run_clean = Run(
            steps=1,
            tokens=10,
            answer="ok",
            trace=[
                Trace(
                    name="memory_tool",
                    args=TraceArgs(action="read"),
                    result="ok",
                )
            ],
        )
        self.assertEqual(layer.check(case, run_clean), [])

        run_forbidden = Run(
            steps=1,
            tokens=10,
            answer="ok",
            trace=[
                Trace(name="calculator_tool", args=TraceArgs(), result="5")
            ],
        )
        fails = layer.check(case, run_forbidden)
        self.assertEqual(len(fails), 1)
        self.assertIn("Forbidden Tool Used: calculator_tool", fails[0])

    def test_custom_eval_layer_open_closed(self):
        class NoProfanityLayer(BaseEvalLayer):
            @property
            def name(self) -> str:
                return "no_profanity"

            def check(self, case: Case, run: Run) -> list[str]:
                if "damn" in run.answer.lower():
                    return ["Profanity detected"]
                return []

        custom_layer = NoProfanityLayer()
        case = Case(name="test", prompt="test")
        run_clean = Run(steps=1, tokens=5, answer="Have a nice day")
        run_dirty = Run(steps=1, tokens=5, answer="Damn it failed")

        self.assertEqual(custom_layer.check(case, run_clean), [])
        self.assertEqual(custom_layer.check(case, run_dirty), ["Profanity detected"])


class TestEvalStorage(unittest.TestCase):
    def test_save_run_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = EvalStorage(Path(tmpdir))
            record = EvalRunRecord(
                run_id="run_test_123",
                timestamp="2026-01-01T00:00:00Z",
                model="test_model",
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                cases=[],
            )
            run_file, latest_file = storage.save_run(record)
            self.assertTrue(run_file.exists())
            self.assertTrue(latest_file.exists())
            self.assertEqual(run_file.name, "run_test_123.json")


if __name__ == "__main__":
    unittest.main()
