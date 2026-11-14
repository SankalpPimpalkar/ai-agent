from typing import Any, Callable, List, Optional, Union
from ollama import Client, Message
from src.agent.conversation.history import ConversationHistory
from src.agent.conversation.prompt_builder import PromptBuilder
from src.agent.core.breakers import LoopBreakers
from src.agent.core.chat import chat_step
from src.agent.core.executor import dispatch_tools, format_single_line
from src.agent.core.types import (
    AgentState,
    ChatClientProtocol,
    Trace,
    TraceArgs,
    to_response,
)
from src.agent.settings import SETTINGS, Settings
from src.agent.tools.builder import load_custom_tools
from src.agent.tools.planner import (
    derive_project_name,
    finalize_plan,
    generate_project_plan,
    init_plan,
    is_project_task,
)
from src.agent.tools.registry import ToolRegistry, create_default_registry
from src.agent.tools.state import STATE


def validate_agent_input(
    user_input: Optional[Union[str, List[Message]]],
    model: Optional[str] = None,
    tool_use: bool = True,
    thinking: bool = False,
    chat_history: Optional[List[Message]] = None,
    kwargs: Optional[dict] = None,
) -> tuple[str, str, bool, bool, List[Message], int]:
    kwargs = kwargs or {}

    if isinstance(user_input, list):
        chat_history = user_input
        user_input = model if isinstance(model, str) else kwargs.get("user_input", "")
        model = tool_use if isinstance(tool_use, str) else kwargs.get("model", None)
        tool_use = (
            thinking
            if isinstance(thinking, bool) and "tool_use" not in kwargs
            else kwargs.get("tool_use", True)
        )
        thinking = kwargs.get("thinking", False)

    if user_input is None:
        user_input = kwargs.get("user_input", "")
    if not isinstance(user_input, str):
        user_input = str(user_input)
    clean_input = user_input.strip()

    history = chat_history if chat_history is not None else []
    if not isinstance(history, list):
        history = list(history)

    if not clean_input and not history:
        raise ValueError("Agent input cannot be empty.")

    validated_model = model or kwargs.get("model") or SETTINGS.DEFAULT_MODEL
    if not isinstance(validated_model, str) or not validated_model.strip():
        validated_model = SETTINGS.DEFAULT_MODEL

    validated_tool_use = bool(
        tool_use if "tool_use" not in kwargs else kwargs["tool_use"]
    )
    validated_thinking = bool(
        thinking if "thinking" not in kwargs else kwargs["thinking"]
    )

    raw_steps = kwargs.get("max_steps") or 25
    try:
        max_steps = int(raw_steps)
        if max_steps <= 0:
            max_steps = 25
        max_steps = min(max_steps, 100)
    except (ValueError, TypeError):
        max_steps = 25

    return (
        clean_input,
        validated_model,
        validated_tool_use,
        validated_thinking,
        history,
        max_steps,
    )


def validate_agent_output(state: Any, last_response: Any = None) -> dict:
    answer = getattr(state, "answer", "") or ""

    if (not answer or answer == "Stopped after too many steps.") and last_response:
        msg = getattr(last_response, "message", None)
        if msg and getattr(msg, "content", None):
            answer = msg.content

    answer = answer.removeprefix("end_").strip()
    if not answer:
        answer = "Task executed successfully."
    state.answer = answer

    result = state.to_dict()
    result["answer"] = answer
    return result


class Agent:
    """
    Coordinates the agent reasoning loop, tool execution, circuit breaking,
    and conversation history.
    """

    def __init__(
        self,
        client: Optional[ChatClientProtocol] = None,
        tool_registry: Optional[ToolRegistry] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        breakers_factory: Optional[Callable[[], LoopBreakers]] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or SETTINGS
        self.client = client or self.settings.OLLAMA_CLIENT
        self.tool_registry = tool_registry or create_default_registry(
            self.settings, self.client
        )
        self.prompt_builder = prompt_builder or PromptBuilder(self.settings)
        self.breakers_factory = breakers_factory or (lambda: LoopBreakers())

    def _log(
        self,
        message: str,
        on_progress: Optional[Callable[[str], None]] = None,
        verbose: bool = False,
    ) -> None:
        if on_progress:
            on_progress(message)
        elif verbose or self.settings.DEV_MODE:
            print(message, flush=True)

    def run(
        self,
        user_input: Optional[Union[str, List[Message]]] = None,
        model: Optional[str] = None,
        tool_use: bool = True,
        thinking: bool = False,
        chat_history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
        tool_instructions: Optional[str] = None,
        memory_instructions: Optional[str] = None,
        max_steps: Optional[int] = None,
        verbose: bool = False,
        on_progress: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> dict:
        user_input, model, tool_use, thinking, history_input, max_iter = (
            validate_agent_input(
                user_input, model, tool_use, thinking, chat_history, kwargs
            )
        )
        is_verbose = verbose or kwargs.get("verbose", False)
        progress_cb = on_progress or kwargs.get("on_progress")

        STATE.reset()
        state = AgentState(_loop_breakers=self.breakers_factory())
        state.logs.append(f"User Input: {user_input}")

        try:
            prompt_text = user_input
            if tool_use and is_project_task(user_input):
                proj = derive_project_name(user_input)
                STATE.current_project = proj
                tasks, tests = generate_project_plan(
                    user_input, proj, client=self.client, model=model
                )
                plan_content, plan_file = init_plan(
                    proj, tasks=tasks, tests=tests, prompt=user_input, client=self.client, model=model
                )
                init_msg = f"📋 Plan: {proj} ({len(tasks)} tasks, {len(tests)} tests)"
                self._log(init_msg, on_progress=progress_cb, verbose=is_verbose)

                prompt_text = (
                    f"{user_input}\n\n"
                    f"### Current PLAN.md ({proj}/PLAN.md):\n"
                    f"{plan_content.strip()}\n\n"
                    f"### Execution Guidelines:\n"
                    f"1. Follow the tasks and tests in order.\n"
                    f"2. Use planner_tool(action='mark_done', task='...') to mark each task and test as completed [x] as you finish them.\n"
                    f"3. Always use project='{proj}' when calling cli_tool.\n"
                    f"4. All file paths must be relative to the project root (e.g. path='main.py', NOT path='artifacts/...').\n"
                    f"5. Run and verify all tests before completing."
                )

            conversation = ConversationHistory(history_input)
            conversation.trim()
            conversation.append(Message(role="user", content=prompt_text))

            last_res = None
            for i in range(max_iter if tool_use else 1):
                state.steps = i + 1

                self._log(f"\n[Step {state.steps}]", on_progress=progress_cb, verbose=is_verbose)

                if tool_use:
                    custom_tools, custom_funcs = load_custom_tools()
                    active_registry = self.tool_registry.merge_custom(
                        custom_tools, custom_funcs
                    )
                    all_tools = active_registry.get_specs()
                    all_funcs = active_registry.get_functions()
                else:
                    all_tools, all_funcs = None, {}

                msgs = self.prompt_builder.build_messages(
                    chat_history=conversation.get_messages(),
                    tool_use=tool_use,
                    system_prompt=system_prompt,
                    tool_instructions=tool_instructions,
                    memory_instructions=memory_instructions,
                )

                res = chat_step(
                    client=self.client,
                    model=model,
                    messages=msgs,
                    tools=all_tools,
                    thinking=thinking,
                    logs=state.logs,
                    step=state.steps,
                )

                if not res or not res.message:
                    state.answer = "Error: Failed to obtain response from model."
                    break

                last_res = res
                conversation.append(res.message)
                state.record_usage(res.prompt_eval_count or 0, res.eval_count or 0)

                if res.message.thinking:
                    think_text = res.message.thinking.strip()
                    state.logs.append(f"[Step {state.steps}] Thinking: {think_text}")
                    think_1line = format_single_line(think_text, max_len=130)
                    self._log(
                        f"  🧠 Thinking: {think_1line}",
                        on_progress=progress_cb,
                        verbose=is_verbose,
                    )

                if res.message.content and not res.message.tool_calls:
                    content_text = res.message.content.strip()
                    state.logs.append(f"[Step {state.steps}] Content: {content_text}")
                    ans_1line = format_single_line(content_text, max_len=130)
                    self._log(
                        f"  💬 Answer: {ans_1line}",
                        on_progress=progress_cb,
                        verbose=is_verbose,
                    )

                if not res.message.tool_calls:
                    content = (res.message.content or "").strip()
                    if not content and state.steps < max_iter:
                        conversation.append(
                            Message(
                                role="user",
                                content="Please proceed with tool calls or your final response.",
                            )
                        )
                        continue

                    has_created = any(
                        t["name"] == "cli_tool"
                        and t["args"].get("action") == "create_file"
                        for t in state.trace
                    )
                    has_run = any(
                        t["name"] == "cli_tool" and t["args"].get("action") == "run"
                        for t in state.trace
                    )
                    if has_created and not has_run and state.steps < max_iter:
                        conversation.append(
                            Message(
                                role="user",
                                content="Please run and verify the created code using cli_tool action='run'.",
                            )
                        )
                        continue

                    state.answer = content
                    state.logs.append(f"[Completed] Final answer: {state.answer}")
                    break

                dispatch_tools(
                    res.message.tool_calls,
                    all_funcs,
                    state,
                    conversation._messages,
                    dev_mode=self.settings.DEV_MODE,
                    verbose=is_verbose,
                    on_progress=progress_cb,
                )

            proj = getattr(STATE, "current_project", "")
            if proj:
                finalize_plan(proj, state)

            return validate_agent_output(state, last_res)

        finally:
            STATE.reset()


def run_agent(
    user_input: Optional[Union[str, List[Message]]] = None,
    model: Optional[str] = None,
    tool_use: bool = True,
    thinking: bool = False,
    chat_history: Optional[List[Message]] = None,
    system_prompt: Optional[str] = None,
    tool_instructions: Optional[str] = None,
    memory_instructions: Optional[str] = None,
    max_steps: Optional[int] = None,
    client: Optional[Client] = None,
    verbose: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> dict:
    active_client = client or kwargs.get("client")
    agent = Agent(client=active_client)
    return agent.run(
        user_input=user_input,
        model=model,
        tool_use=tool_use,
        thinking=thinking,
        chat_history=chat_history,
        system_prompt=system_prompt,
        tool_instructions=tool_instructions,
        memory_instructions=memory_instructions,
        max_steps=max_steps,
        verbose=verbose,
        on_progress=on_progress,
        **kwargs,
    )
