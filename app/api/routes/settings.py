from fastapi import APIRouter, Depends, HTTPException, status
from app.agent.security import validate_user_instructions
from app.api.deps import get_settings
from app.config import Settings
from app.schemas.settings import SettingsResponse, SettingsUpdateRequest

router = APIRouter(prefix="/settings", tags=["settings"])


def _build_settings_response(settings: Settings) -> SettingsResponse:
    return SettingsResponse(
        app_name=settings.APP_NAME,
        model=settings.MODEL,
        temperature=settings.TEMPERATURE,
        max_iterations=settings.MAX_ITERATIONS,
        thinking=getattr(settings, "ENABLE_REASONING", False),
        instructions=getattr(settings, "USER_INSTRUCTIONS", None),
        rag_enabled=settings.ENABLE_RAG,
        tools_enabled=settings.ENABLE_TOOLS,
        mcp_enabled=settings.ENABLE_MCP,
        persistent_memory_enabled=settings.ENABLE_PERSISTENT_MEMORY,
    )


@router.get("", response_model=SettingsResponse)
async def get_settings_route(settings: Settings = Depends(get_settings)):
    """
    Retrieves current agent configuration, user instructions,
    sampling temperature, max iterations, and active feature flags.
    The system prompt is internal and not exposed over the API.
    """
    return _build_settings_response(settings)


@router.post("", response_model=SettingsResponse)
@router.put("", response_model=SettingsResponse)
@router.patch("", response_model=SettingsResponse)
async def update_settings_route(
    request: SettingsUpdateRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Configures and updates agent settings including user instructions, temperature,
    max iterations, and thinking mode. The system prompt is immutable and read-only.
    Validates user instructions to reject potential prompt injections or system overrides.
    """
    if request.reset_to_default:
        settings.USER_INSTRUCTIONS = None
    else:
        if request.instructions is not None:
            valid_inst = validate_user_instructions(request.instructions)
            settings.USER_INSTRUCTIONS = valid_inst

    if request.temperature is not None:
        settings.TEMPERATURE = request.temperature

    if request.max_iterations is not None:
        settings.MAX_ITERATIONS = request.max_iterations

    if request.thinking is not None:
        settings.ENABLE_REASONING = request.thinking

    return _build_settings_response(settings)
