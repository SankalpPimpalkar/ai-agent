from typing import List
from .schemas import Trace, Case, Run

BUILTIN_TOOLS = {"tool_builder", "memory_tool", "cli_tool", "calculator_tool"}

def _label(trace: Trace) -> str:
    actn = trace.args.action or None
    return f"{trace.name}:{actn}" if actn else trace.name

def _hit(step: str, label: str) -> bool:
    if label == step or label.startswith(step + ":") or label.startswith(step + "_") or step.startswith(label + "_"):
        return True
    base_step = step.split(":")[0]
    base_label = label.split(":")[0]
    if base_step not in BUILTIN_TOOLS and base_label not in BUILTIN_TOOLS:
        if base_step in base_label or base_label in base_step:
            return True
    return False

def tool_eval(case: Case, run: Run) -> List[str]:
    """Check that no forbidden tools were used during execution."""
    labels = [_label(t) for t in run.trace]
    fails = []
    for t in case.forbid_tools:
        if any(_hit(t, l) for l in labels):
            fails.append(f"Forbidden Tool Used: {t}")
    return fails

def output_eval(case: Case, run: Run) -> List[str]:
    """Verify that expected outputs or substrings are present in the final response."""
    fails = []
    for expected in case.expected_output:
        if expected.lower() not in run.answer.lower():
            fails.append(f"Expected output missing: '{expected}'")
    return fails

def process_eval(case: Case, run: Run) -> List[str]:
    """Verify that the execution completed within the maximum allowed step count."""
    fails = []
    if run.steps > case.max_steps:
        fails.append(f"Too many steps: {run.steps} (max: {case.max_steps})")
    return fails

EVAL_LAYERS = {
    'tool_eval': tool_eval,
    'output_eval': output_eval,
    'process_eval': process_eval,
}
