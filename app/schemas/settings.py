from typing import Optional
from pydantic import BaseModel, Field


class SettingsUpdateRequest(BaseModel):
    instructions: Optional[str] = Field(
        default=None,
        description="Global user instructions to steer agent behavior, tone, and task execution. Must not contain prompt injection or attempt to override security rules and core capabilities.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Default sampling temperature (0.0 to 2.0).",
    )
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Default maximum tool loop iterations (1 to 50).",
    )
    thinking: Optional[bool] = Field(
        default=None,
        description="Default thinking / reasoning mode for the agent.",
    )
    reset_to_default: bool = Field(
        default=False,
        description="If true, clears custom user instructions and resets runtime settings to default.",
    )


class SettingsResponse(BaseModel):
    app_name: str
    model: str
    temperature: float
    max_iterations: int
    thinking: bool
    instructions: Optional[str] = None
    rag_enabled: bool
    tools_enabled: bool
    mcp_enabled: bool
    persistent_memory_enabled: bool
