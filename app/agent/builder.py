from typing import Any, AsyncGenerator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.agent.llm import get_llm
from app.config import settings
from app.tools.registry import get_all_tools


class AgentRunner:
    """
    Executes conversational turns using an LLM optionally bound with tools
    from local, RAG, and MCP sources.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
    ):
        self._llm = llm
        self._tools = tools

    async def get_tools(self) -> List[Any]:
        if self._tools is not None:
            return self._tools
        return await get_all_tools()

    async def get_runnable(self, model: Optional[str] = None) -> Any:
        active_llm = self._llm or get_llm(model=model)
        active_tools = await self.get_tools()
        if active_tools and hasattr(active_llm, "bind_tools"):
            return active_llm.bind_tools(active_tools)
        return active_llm

    def build_messages(
        self,
        message: str,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        sys = system_prompt or settings.SYSTEM_PROMPT
        msgs: List[BaseMessage] = [SystemMessage(content=sys)]
        if history:
            msgs.extend(history)
        msgs.append(HumanMessage(content=message))
        return msgs

    async def execute(
        self,
        message: str,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
    ) -> str:
        """
        Invokes the model, automatically handles tool calling iterations if
        tools are invoked, and returns the final assistant answer.
        """
        tools = await self.get_tools()
        tools_map = {t.name: t for t in tools} if tools else {}

        active_llm = self._llm or get_llm(model=model)
        runnable = (
            active_llm.bind_tools(tools)
            if tools and hasattr(active_llm, "bind_tools")
            else active_llm
        )

        messages = self.build_messages(
            message=message, history=history, system_prompt=system_prompt
        )

        for _ in range(max_iterations):
            response = await runnable.ainvoke(messages)
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                content = getattr(response, "content", "")
                if isinstance(content, list):
                    content = "".join(str(part) for part in content)
                return str(content)

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

        last_msg = messages[-1]
        return str(getattr(last_msg, "content", "")) or "Task completed."

    async def execute_stream(
        self,
        message: str,
        history: Optional[List[BaseMessage]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Streams generated tokens from the runnable."""
        runnable = await self.get_runnable(model=model)
        messages = self.build_messages(
            message=message, history=history, system_prompt=system_prompt
        )
        async for chunk in runnable.astream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(str(part) for part in content)
            if content:
                yield str(content)


async def build_agent(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List[Any]] = None,
) -> AgentRunner:
    """Assembles and returns an AgentRunner instance."""
    return AgentRunner(llm=llm, tools=tools)
