from typing import Any, Callable, Dict, List, Optional
from ollama import Message
from src.agent.core.types import AgentState
from src.agent.tools.planner import mark_next_task_done
from src.agent.tools.state import STATE


def format_single_line(text: Any, max_len: int = 120) -> str:
    """Collapses newlines and multiple spaces into a clean single line."""
    if text is None:
        return ""
    clean = " ".join(str(text).split()).strip()
    if len(clean) > max_len:
        return clean[: max_len - 3] + "..."
    return clean


def format_tool_call_args(name: str, args: dict) -> str:
    """Formats tool call arguments into a compact single line without dumping large code/content."""
    compact = {}
    for k, v in args.items():
        if k in ("content", "file_content", "text") and isinstance(v, str) and len(v) > 40:
            compact[k] = f"<{len(v)} chars>"
        elif isinstance(v, str) and len(v) > 50:
            compact[k] = v[:47] + "..."
        else:
            compact[k] = v
    arg_str = ", ".join(f"{k}={repr(v)}" for k, v in compact.items())
    return f"{name}({arg_str})"


def dispatch_tools(
    tool_calls: list,
    all_funcs: Dict[str, Callable],
    state: AgentState,
    history: List[Message],
    dev_mode: bool = False,
    verbose: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
):
    """
    Executes tool calls, checks loop breakers, updates history, logs progress,
    and advances plan items when appropriate.
    """
    for tc in tool_calls:
        name = tc.function.name
        args = tc.function.arguments or {}
        call_summary = format_tool_call_args(name, args)
        call_log = f"  🔧 Tool: {call_summary}"
        state.logs.append(f"[Step {state.steps}] Tool Call: {name}({args})")
        if on_progress:
            on_progress(call_log)
        elif dev_mode or verbose:
            print(call_log, flush=True)

        if name in all_funcs:
            try:
                result = all_funcs[name](**args)
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = f"Error: unknown tool '{name}'"

        result = str(result)
        out_summary = format_single_line(result, max_len=130)
        out_log = f"  📋 Output: {out_summary}"
        state.logs.append(f"[Step {state.steps}] Tool Result: {result}")
        if on_progress:
            on_progress(out_log)
        elif dev_mode or verbose:
            print(out_log, flush=True)

        # Check circuit breakers for consecutive errors and repeated calls
        result = state.check_breakers(name, args, result, state.steps)

        # Automatic plan updates upon successful actions
        if name == "cli_tool" and not str(result).startswith("Error"):
            proj = getattr(STATE, "current_project", "") or args.get("project", "")
            if proj:
                action = args.get("action", "")
                if action == "create_file":
                    ok, item = mark_next_task_done(proj, section="tasks")
                    if ok:
                        item_summary = format_single_line(item, max_len=100)
                        plan_log = f"  ✅ Plan: Completed '{item_summary}'"
                        state.logs.append(
                            f"[Step {state.steps}] Plan Update: Completed task '{item}' in {proj}/PLAN.md"
                        )
                        if on_progress:
                            on_progress(plan_log)
                        elif dev_mode or verbose:
                            print(plan_log, flush=True)
                elif action == "run":
                    ok, item = mark_next_task_done(proj, section="tests")
                    if not ok:
                        ok, item = mark_next_task_done(proj, section="tasks")
                    if ok:
                        item_summary = format_single_line(item, max_len=100)
                        plan_log = f"  ✅ Plan: Completed '{item_summary}'"
                        state.logs.append(
                            f"[Step {state.steps}] Plan Update: Completed test '{item}' in {proj}/PLAN.md"
                        )
                        if on_progress:
                            on_progress(plan_log)
                        elif dev_mode or verbose:
                            print(plan_log, flush=True)
        elif name == "planner_tool" and args.get("action") in ("mark_done", "complete") and not str(result).startswith("Error"):
            plan_1line = format_single_line(result, max_len=110)
            plan_log = f"  ✅ Plan: {plan_1line}"
            if on_progress:
                on_progress(plan_log)
            elif dev_mode or verbose:
                print(plan_log, flush=True)

        state.trace.append(
            {"name": name, "args": dict(args), "result": result, "step": state.steps}
        )
        history.append({"role": "tool", "tool_name": name, "content": result})
