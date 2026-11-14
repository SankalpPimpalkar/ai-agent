import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


def _expand_env_vars(val: Any) -> Any:
    """Recursively expands ${VAR} or $VAR environment variables in strings, lists, and dicts."""
    if isinstance(val, str):
        return os.path.expandvars(val)
    elif isinstance(val, list):
        return [_expand_env_vars(item) for item in val]
    elif isinstance(val, dict):
        return {k: _expand_env_vars(v) for k, v in val.items()}
    return val


class MCPServerConfig(BaseModel):
    """
    Standardized configuration for an individual Model Context Protocol (MCP) server.
    Compatible with Claude Desktop, Cursor, and standard MCP client formats.
    """
    command: Optional[str] = Field(
        default=None,
        description="Executable command for stdio transport (e.g. python, uvx, npx)",
    )
    args: List[str] = Field(
        default_factory=list,
        description="Command-line arguments passed to the server executable",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables passed to the subprocess",
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory for the subprocess",
    )
    url: Optional[str] = Field(
        default=None,
        description="Server URL for SSE or HTTP transports",
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional HTTP headers for SSE / HTTP transports",
    )
    transport: Optional[Literal["stdio", "sse", "streamable_http", "websocket"]] = Field(
        default=None,
        description="Transport type. Inferred automatically from command or url if omitted",
    )
    disabled: bool = Field(
        default=False,
        description="Set to true to skip loading this server without deleting its config",
    )

    @model_validator(mode="after")
    def validate_and_infer_transport(self):
        # Auto-detect transport if not explicitly set
        if not self.transport:
            if self.command:
                self.transport = "stdio"
            elif self.url:
                self.transport = "sse"
            else:
                raise ValueError("MCP server must provide either 'command' (stdio) or 'url' (sse)")

        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server must specify a 'command'")

        if self.transport in ("sse", "streamable_http", "websocket") and not self.url:
            raise ValueError(f"{self.transport} MCP server must specify a 'url'")

        return self

    def to_connection_dict(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Converts this configuration into a connection dictionary for MultiServerMCPClient."""
        project_root = base_dir or Path.cwd()

        if self.transport == "stdio":
            cmd = _expand_env_vars(self.command or "")
            # Resolve relative Python commands to current venv python
            if cmd in ("python", "python3"):
                cmd = sys.executable
            else:
                cmd = shutil.which(cmd) or cmd

            # Resolve relative arguments (e.g. repository path ".")
            resolved_args = []
            for arg in self.args:
                expanded = _expand_env_vars(arg)
                if expanded == ".":
                    resolved_args.append(str(project_root))
                elif expanded.startswith("./"):
                    resolved_args.append(str(project_root / expanded[2:]))
                else:
                    resolved_args.append(expanded)

            # Build environment with current environment inheritance
            expanded_env = {k: _expand_env_vars(v) for k, v in self.env.items()}
            full_env = {**os.environ, **expanded_env}

            conn = {
                "transport": "stdio",
                "command": cmd,
                "args": resolved_args,
                "env": full_env,
            }
            if self.cwd:
                conn["cwd"] = str(_expand_env_vars(self.cwd))
            return conn

        # SSE / HTTP transport
        conn = {
            "transport": self.transport,
            "url": _expand_env_vars(self.url or ""),
        }
        if self.headers:
            conn["headers"] = {k: _expand_env_vars(v) for k, v in self.headers.items()}
        return conn


class MCPConfigFile(BaseModel):
    """
    Standardized multi-server configuration file model.
    Accepts both:
      1. {"mcpServers": { "name": { ... } }} (Claude Desktop / Cursor standard)
      2. { "name": { ... } } (Flat dictionary format)
    """
    mcpServers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    @classmethod
    def from_raw_data(cls, data: Any) -> "MCPConfigFile":
        if not isinstance(data, dict):
            raise ValueError("MCP configuration file must contain a JSON object")

        # Standard format with root "mcpServers" key
        if "mcpServers" in data and isinstance(data["mcpServers"], dict):
            return cls(mcpServers={
                k: MCPServerConfig.model_validate(v) for k, v in data["mcpServers"].items()
            })

        # Flat dictionary format where top-level keys are server names
        servers = {}
        for k, v in data.items():
            if isinstance(v, dict):
                servers[k] = MCPServerConfig.model_validate(v)
        return cls(mcpServers=servers)
