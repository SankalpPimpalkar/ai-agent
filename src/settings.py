from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cached_property
from ollama import Client
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    DEV_MODE: bool = True

    DEFAULT_MODEL: str = "gemma4:12b"
    MODEL_CTX_SZ: int = 8192
    THINKING: bool = True

    OLLAMA_API_KEY: str
    OLLAMA_WEB_SEARCH: bool = False

    ARTIFACTS_FOLDER_NAME: str = "artifacts"
    MEMORY_FOLDER_NAME: str = "memory"
    AGENT_TOOLS_FOLDER_NAME: str = "agent-tools"

    SYSTEM_PROMPT: str = Field(default_factory=lambda: (BASE_DIR / "system-prompt.txt").read_text())

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    @cached_property
    def OLLAMA_CLIENT(self) -> Client:
        return Client(headers={"Authorization": f"Bearer {self.OLLAMA_API_KEY}"})

SETTINGS = Settings()