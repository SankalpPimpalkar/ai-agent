from typing import List
from evals.layers.base import BaseEvalLayer, _hit, _label
from evals.models import Case, Run


class ToolEvalLayer(BaseEvalLayer):
    """Checks that no forbidden tools were invoked during the execution."""

    @property
    def name(self) -> str:
        return "tool_eval"

    def check(self, case: Case, run: Run) -> List[str]:
        labels = [_label(t) for t in run.trace]
        fails = []
        for t in case.forbid_tools:
            if any(_hit(t, l) for l in labels):
                fails.append(f"Forbidden Tool Used: {t}")
        return fails
