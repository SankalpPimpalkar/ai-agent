from .schemas import Trace, Case, Run

def _label(trace: Trace):
    actn = trace.args.action or None
    return f"{trace.name}:{actn}" if actn else trace.name

def _hit(step, label):
    return label == step or label.startswith(step + ":")

def tool_eval(case: Case, run: Run):
    labels = [_label(t) for t in run.trace]
    it, fails = iter(labels), []

    for step in case.tool_order:
        if not any(_hit(step, l) for l in it):
            fails.append(f"Tool Order Mismatch: {step}")
            break

    for t in case.forbid_tools:
        if any(_hit(t, l) for l in labels):
            fails.append(f"Forbidden Tool Used: {t}")
    return fails

def process_eval(case: Case, run: Run):
    fails = []
    if run.steps > case.max_steps:
        fails.append(f"Too many steps: {run.steps}")
    return fails

EVAL_LAYERS = {
    'tool_eval': tool_eval,
    'process_eval': process_eval
}