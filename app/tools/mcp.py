from typing import Any, List, Optional
from app.config import settings


async def get_mcp_tools(server_configs: Optional[List[str]] = None) -> List[Any]:
    """
    Connects to external Model Context Protocol (MCP) servers using
    langchain-mcp-adapters and converts remote tools into LangChain tools.
    """
    servers = server_configs or settings.MCP_SERVERS
    if not settings.ENABLE_MCP or not servers:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
        client = MultiServerMCPClient()
        for server in servers:
            await client.connect_to_server(server)
        return client.get_tools()
    except ImportError:
        # langchain-mcp-adapters is optional; graceful fallback if not installed
        return []
    except Exception as e:
        # Return empty list on connection error to ensure resilient startup
        print(f"[MCP] Warning: Failed to connect to MCP servers: {e}")
        return []
