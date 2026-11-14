import uuid
from typing import Any, AsyncGenerator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.agent.llm import get_llm
from app.agent.memory import load_session_history, save_session_history
from app.agent.middleware import (
    prune_old_tool_outputs_middleware,
    summarize_context_middleware,
)
from app.config import settings
from app.tools.registry import get_all_tools


class AgentRunner:
    """
    Executes conversational turns using an LLM optionally bound with tools
    from local, RAG, and MCP sources, with persistent memory and middleware support.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._llm = llm
        self._tools = tools
        self._checkpointer = checkpointer
        self.last_session_id: Optional[str] = None

    async def get_tools(self) -> List[Any]:
        if self._tools is not None:
            return self._tools
        return await get_all_tools()

    async def get_runnable(self, model: Optional[str] = None) -> Any:
        active_llm = self._llm or get_llm(model=model)
        active_tools = await self.get_tools()
        if active_tools and hasattr(active_llm, "bind_tools"):
            try:
                return active_llm.bind_tools(active_tools)
            except (NotImplementedError, AttributeError):
                return active_llm
        return active_llm

    def build_messages(
        self,
        message: str,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        sys = system_prompt or settings.SYSTEM_PROMPT
        msgs: List[BaseMessage] = []
        if history:
            if isinstance(history[0], SystemMessage):
                msgs.extend(history)
            else:
                msgs.append(SystemMessage(content=sys))
                msgs.extend(history)
        else:
            msgs.append(SystemMessage(content=sys))
        msgs.append(HumanMessage(content=message))
        return msgs

    async def execute(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
    ) -> str:
        """
        Invokes the model, automatically handles tool calling iterations if
        tools are invoked, applies persistent memory and middlewares, and returns the final assistant answer.
        """
        active_session_id = session_id or str(uuid.uuid4())
        self.last_session_id = active_session_id

        prior_history: List[BaseMessage] = []
        if settings.ENABLE_PERSISTENT_MEMORY:
            persisted = await load_session_history(
                active_session_id, checkpointer=self._checkpointer
            )
            if persisted:
                prior_history.extend(persisted)
        if history:
            prior_history.extend(history)

        tools = await self.get_tools()
        tools_map = {t.name: t for t in tools} if tools else {}

        active_llm = self._llm or get_llm(model=model)
        runnable = active_llm
        if tools and hasattr(active_llm, "bind_tools"):
            try:
                runnable = active_llm.bind_tools(tools)
            except (NotImplementedError, AttributeError):
                runnable = active_llm

        messages = self.build_messages(
            message=message, history=prior_history, system_prompt=system_prompt
        )

        # Apply middlewares:
        # 1. Prune old tool outputs to placeholders
        messages = prune_old_tool_outputs_middleware(messages)
        # 2. Summarize context if size overflows
        messages = await summarize_context_middleware(messages, llm=active_llm)

        final_content = "Task completed."

        for _ in range(max_iterations):
            response = await runnable.ainvoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                content = getattr(response, "content", "")
                if isinstance(content, list):
                    content = "".join(str(part) for part in content)
                final_content = str(content)
                break

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id") or tool_name

                if tool_name in tools_map:
                    try:
                        tool_result = await tools_map[tool_name].ainvoke(tool_args)
                    except Exception as err:
                        tool_result = f"Error executing tool {tool_name}: {err}"
                else:
                    tool_result = f"Error: unknown tool '{tool_name}'"

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

        if settings.ENABLE_PERSISTENT_MEMORY:
            await save_session_history(
                active_session_id, messages, checkpointer=self._checkpointer
            )

        return final_content

    async def execute_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
        stream_tools: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated tokens from the LLM, automatically executing tool calls
        in a loop, managing persistent memory and context compression.
        """
        active_session_id = session_id or str(uuid.uuid4())
        self.last_session_id = active_session_id

        prior_history: List[BaseMessage] = []
        if settings.ENABLE_PERSISTENT_MEMORY:
            persisted = await load_session_history(
                active_session_id, checkpointer=self._checkpointer
            )
            if persisted:
                prior_history.extend(persisted)
        if history:
            prior_history.extend(history)

        tools = await self.get_tools()
        tools_map = {t.name: t for t in tools} if tools else {}

        active_llm = self._llm or get_llm(model=model)
        runnable = active_llm
        if tools and hasattr(active_llm, "bind_tools"):
            try:
                runnable = active_llm.bind_tools(tools)
            except (NotImplementedError, AttributeError):
                runnable = active_llm

        messages = self.build_messages(
            message=message, history=prior_history, system_prompt=system_prompt
        )

        # Apply middlewares:
        messages = prune_old_tool_outputs_middleware(messages)
        messages = await summarize_context_middleware(messages, llm=active_llm)

        for _ in range(max_iterations):
            accumulated: Optional[AIMessageChunk] = None

            async for chunk in runnable.astream(messages):
                if isinstance(chunk, str):
                    chunk = AIMessageChunk(content=chunk)
                elif not isinstance(chunk, AIMessageChunk) and isinstance(chunk, BaseMessage):
                    chunk = AIMessageChunk(
                        content=chunk.content,
                        tool_calls=getattr(chunk, "tool_calls", None) or [],
                        additional_kwargs=getattr(chunk, "additional_kwargs", {}),
                    )

                if accumulated is None:
                    accumulated = chunk
                else:
                    accumulated = accumulated + chunk

                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(str(part) for part in content)
                if content:
                    yield str(content)

            if accumulated is None:
                break

            ai_message = AIMessage(
                content=accumulated.content,
                tool_calls=getattr(accumulated, "tool_calls", None) or [],
                additional_kwargs=getattr(accumulated, "additional_kwargs", {}),
            )
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", None)
            if not tool_calls:
                break

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id") or tool_name

                if stream_tools:
                    yield f"\n[Calling tool '{tool_name}'...]\n"

                if tool_name in tools_map:
                    try:
                        tool_result = await tools_map[tool_name].ainvoke(tool_args)
                    except Exception as err:
                        tool_result = f"Error executing tool {tool_name}: {err}"
                else:
                    tool_result = f"Error: unknown tool '{tool_name}'"

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

        if settings.ENABLE_PERSISTENT_MEMORY:
            await save_session_history(
                active_session_id, messages, checkpointer=self._checkpointer
            )


async def build_agent(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[Any]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> AgentRunner:
    """Assembles and returns an AgentRunner instance."""
    return AgentRunner(llm=llm, tools=tools, checkpointer=checkpointer)
