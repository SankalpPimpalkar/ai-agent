from typing import List
from evals.layers.base import BaseEvalLayer
from evals.models import Case, Run


class OutputEvalLayer(BaseEvalLayer):
    """Verifies that expected output substrings appear in the final answer."""

    @property
    def name(self) -> str:
        return "output_eval"

    def check(self, case: Case, run: Run) -> List[str]:
        fails = []
        for expected in case.expected_output:
            if expected.lower() not in run.answer.lower():
                fails.append(f"Expected output missing: '{expected}'")
        return fails
