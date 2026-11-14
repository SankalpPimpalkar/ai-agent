from typing import List
from evals.layers.base import BaseEvalLayer
from evals.models import Case, Run


class ProcessEvalLayer(BaseEvalLayer):
    """Verifies that the run stayed within the maximum step count."""

    @property
    def name(self) -> str:
        return "process_eval"

    def check(self, case: Case, run: Run) -> List[str]:
        fails = []
        if run.steps > case.max_steps:
            fails.append(f"Too many steps: {run.steps} (max: {case.max_steps})")
        return fails
