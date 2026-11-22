import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import BASE_DIR, settings
from app.schemas.mcp import MCPConfigFile


def load_mcp_config_file(config_path: Optional[str] = None) -> Optional[MCPConfigFile]:
    """
    Locates, reads, and validates the standardized MCP server configuration file.
    Checks config_path, then falls back to mcp_servers.json, mcp.json.
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([
        Path(settings.MCP_CONFIG_PATH),
        settings.BASE_DIR / settings.MCP_CONFIG_PATH,
        settings.BASE_DIR / "mcp_servers.json",
        settings.BASE_DIR / "mcp.json",
    ])

    resolved_file: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            resolved_file = candidate
            break

    if not resolved_file:
        return None

    try:
        raw_text = resolved_file.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        return MCPConfigFile.from_raw_data(data)
    except Exception as e:
        print(f"[MCP] Error loading MCP configuration from {resolved_file}: {e}")
        return None


def get_mcp_connections(config_file: Optional[MCPConfigFile] = None) -> Dict[str, Any]:
    """
    Converts validated MCP server configurations into connection dictionaries
    compatible with MultiServerMCPClient, skipping disabled servers.
    """
    cfg = config_file or load_mcp_config_file()
    if not cfg or not cfg.mcpServers:
        return {}

    connections = {}
    for name, server in cfg.mcpServers.items():
        if server.disabled:
            continue
        try:
            connections[name] = server.to_connection_dict(base_dir=settings.BASE_DIR)
        except Exception as e:
            print(f"[MCP] Warning: Skipping invalid server configuration '{name}': {e}")

    return connections


async def _load_single_server_tools(
    server_name: str,
    server_conn: Dict[str, Any],
    client_cls: Any,
) -> List[Any]:
    """
    Loads tools from a single MCP server with isolated error handling and a timeout
    so an invalid, broken, or hanging server never prevents other servers from working.
    """
    try:
        client = client_cls(connections={server_name: server_conn})
        return await asyncio.wait_for(client.get_tools(), timeout=5.0)
    except asyncio.TimeoutError:
        print(f"[MCP] Warning: Timed out waiting for MCP server '{server_name}' to return tools.")
        return []
    except Exception as e:
        print(f"[MCP] Warning: Failed to load tools from MCP server '{server_name}': {e}")
        return []


async def get_mcp_tools(config_path: Optional[str] = None) -> List[Any]:
    """
    Automatically connects to all valid MCP servers configured in mcp_servers.json
    using langchain-mcp-adapters and returns them as LangChain BaseTool instances.
    Each server is connected with isolated error boundaries so failing or misconfigured
    servers never drop tools from healthy servers.
    """
    if not settings.ENABLE_MCP:
        return []

    cfg = load_mcp_config_file(config_path)
    connections = get_mcp_connections(cfg)
    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        print(f"[MCP] Warning: langchain-mcp-adapters is not installed: {e}")
        return []

    results = await asyncio.gather(
        *[_load_single_server_tools(name, conn, MultiServerMCPClient) for name, conn in connections.items()]
    )

    tools: List[Any] = []
    for server_tools in results:
        tools.extend(server_tools)

    return tools
