from functools import cached_property
from pathlib import Path
from typing import Optional
from ollama import Client
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_prompt(name: str) -> str:
    prompt_file = PROMPTS_DIR / name
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return ""


class DatabaseConfig(BaseModel):
    url: str = Field(default_factory=lambda: f"sqlite+aiosqlite:///{DATA_DIR / 'app.db'}")


class ModelConfig(BaseModel):
    default_model: str = "gemma4:12b"
    context_size: int = 32768
    thinking: bool = False
    api_key: str = ""
    web_search: bool = False


class PathConfig(BaseModel):
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    artifacts_dir: Path = DATA_DIR / "artifacts"
    artifacts_folder: str = "artifacts"
    memory_dir: Path = DATA_DIR / "memory"
    tools_dir: Path = DATA_DIR / "agent-tools"

class PromptConfig(BaseModel):
    system_prompt: str = Field(
        default_factory=lambda: _read_prompt("system.txt")
    )
    tool_instructions: str = Field(
        default_factory=lambda: _read_prompt("tools.txt")
    )
    memory_instructions: str = Field(
        default_factory=lambda: _read_prompt("memory.txt")
    )
    plan_instructions: str = Field(
        default_factory=lambda: _read_prompt("plan.txt")
    )


class Settings(BaseSettings):
    DEV_MODE: bool = False
    DATABASE_URL: str = Field(
        default_factory=lambda: f"sqlite+aiosqlite:///{DATA_DIR / 'app.db'}"
    )

    DEFAULT_MODEL: str = "gemma4:12b"
    MODEL_CTX_SZ: int = 32768
    THINKING: bool = False

    OLLAMA_API_KEY: str = ""
    OLLAMA_WEB_SEARCH: bool = False

    ARTIFACTS_FOLDER_NAME: str = str(DATA_DIR / "artifacts")
    MEMORY_FOLDER_NAME: str = str(DATA_DIR / "memory")
    AGENT_TOOLS_FOLDER_NAME: str = str(DATA_DIR / "agent-tools")

    SYSTEM_PROMPT: str = Field(
        default_factory=lambda: _read_prompt("system.txt")
    )
    TOOL_INSTRUCTIONS: str = Field(
        default_factory=lambda: _read_prompt("tools.txt")
    )
    MEMORY_INSTRUCTIONS: str = Field(
        default_factory=lambda: _read_prompt("memory.txt")
    )
    PLAN_INSTRUCTIONS: str = Field(
        default_factory=lambda: _read_prompt("plan.txt")
    )

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    @cached_property
    def db(self) -> DatabaseConfig:
        return DatabaseConfig(url=self.DATABASE_URL)

    @cached_property
    def model(self) -> ModelConfig:
        return ModelConfig(
            default_model=self.DEFAULT_MODEL,
            context_size=self.MODEL_CTX_SZ,
            thinking=self.THINKING,
            api_key=self.OLLAMA_API_KEY,
            web_search=self.OLLAMA_WEB_SEARCH,
        )

    @cached_property
    def paths(self) -> PathConfig:
        return PathConfig()

    @cached_property
    def prompts(self) -> PromptConfig:
        return PromptConfig(
            system_prompt=self.SYSTEM_PROMPT,
            tool_instructions=self.TOOL_INSTRUCTIONS,
            memory_instructions=self.MEMORY_INSTRUCTIONS,
            plan_instructions=self.PLAN_INSTRUCTIONS,
        )

    def create_client(self, api_key: Optional[str] = None) -> Client:
        key = api_key or self.OLLAMA_API_KEY
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return Client(headers=headers)

    @cached_property
    def OLLAMA_CLIENT(self) -> Client:
        return self.create_client()


SETTINGS = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "memory").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "agent-tools").mkdir(parents=True, exist_ok=True)
