from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    APP_NAME: str = "ai-agent"
    BASE_DIR: Path = BASE_DIR
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL: str = "gemma4:12b"
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = "You are a helpful and versatile AI assistant."

    # Database settings
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DATA_DIR / 'app.db'}"

    # Feature flags
    ENABLE_RAG: bool = False
    ENABLE_TOOLS: bool = False
    ENABLE_MCP: bool = False
    ENABLE_PERSISTENT_MEMORY: bool = True

    # Memory and checkpointer settings
    CHECKPOINTER_DB_PATH: str = str(DATA_DIR / "app.db")
    MAX_CONTEXT_CHARACTERS: int = 8000
    MAX_CONTEXT_MESSAGES: int = 20
    SUMMARY_KEEP_RECENT: int = 4

    # RAG settings
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # MCP configuration file path (standard mcpServers JSON)
    MCP_CONFIG_PATH: str = "mcp_servers.json"


settings = Settings()
