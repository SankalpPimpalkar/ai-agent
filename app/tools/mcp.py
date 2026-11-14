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


async def get_mcp_tools(config_path: Optional[str] = None) -> List[Any]:
    """
    Automatically connects to all valid MCP servers configured in mcp_servers.json
    using langchain-mcp-adapters and returns them as LangChain BaseTool instances.
    """
    if not settings.ENABLE_MCP:
        return []

    cfg = load_mcp_config_file(config_path)
    connections = get_mcp_connections(cfg)
    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(connections=connections)
        tools = await client.get_tools()
        return tools
    except ImportError as e:
        print(f"[MCP] Warning: langchain-mcp-adapters is not installed: {e}")
        return []
    except Exception as e:
        print(f"[MCP] Error connecting to MCP servers: {e}")
        return []
