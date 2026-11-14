from abc import ABC, abstractmethod
from typing import List
from evals.models import Case, Run, Trace

BUILTIN_TOOLS = {"tool_builder", "memory_tool", "cli_tool", "calculator_tool", "cache_tool", "planner_tool"}


def _label(trace: Trace) -> str:
    actn = trace.args.action or None
    return f"{trace.name}:{actn}" if actn else trace.name


def _hit(step: str, label: str) -> bool:
    if (
        label == step
        or label.startswith(step + ":")
        or label.startswith(step + "_")
        or step.startswith(label + "_")
    ):
        return True
    base_step = step.split(":")[0]
    base_label = label.split(":")[0]
    if base_step not in BUILTIN_TOOLS and base_label not in BUILTIN_TOOLS:
        if base_step in base_label or base_label in base_step:
            return True
    return False


class BaseEvalLayer(ABC):
    """
    Abstract base class for all evaluation layers.
    Follows Open/Closed Principle: new eval checks are implemented by subclassing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this eval layer."""
        ...

    @abstractmethod
    def check(self, case: Case, run: Run) -> List[str]:
        """
        Evaluates the case against the execution run.
        Returns a list of failure descriptions, or an empty list if passed.
        """
        ...
