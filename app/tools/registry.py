from typing import Any, List
from app.config import settings
from app.tools.local import LOCAL_TOOLS
from app.tools.mcp import get_mcp_tools
from app.tools.rag import RAG_TOOLS


async def get_all_tools(
    include_local: bool = True,
    include_rag: bool = True,
    include_mcp: bool = True,
) -> List[Any]:
    """
    Collects and returns all active tools from local, RAG, and MCP sources
    based on configuration flags.
    """
    tools: List[Any] = []

    if include_local and settings.ENABLE_TOOLS:
        tools.extend(LOCAL_TOOLS)

    if include_rag and settings.ENABLE_RAG:
        tools.extend(RAG_TOOLS)

    if include_mcp and settings.ENABLE_MCP:
        mcp_tools = await get_mcp_tools()
        tools.extend(mcp_tools)

    return tools
