from typing import Any, AsyncGenerator, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from app.agent.llm import get_llm
from app.config import settings
from app.tools.registry import get_all_tools


class AgentRunner:
    """
    Executes conversational turns using an LLM optionally bound with tools.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Any]] = None,
    ):
        self._llm = llm
        self._tools = tools

    async def get_runnable(self, model: Optional[str] = None) -> Any:
        active_llm = self._llm or get_llm(model=model)
        active_tools = self._tools if self._tools is not None else await get_all_tools()
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
    ) -> str:
        """Invokes the runnable and returns the generated content."""
        runnable = await self.get_runnable(model=model)
        messages = self.build_messages(
            message=message, history=history, system_prompt=system_prompt
        )
        response = await runnable.ainvoke(messages)
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        return str(content)

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
