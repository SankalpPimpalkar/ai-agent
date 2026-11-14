from datetime import datetime, timezone
import math
from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Returns the current UTC date and time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@tool
def calculate_expression(expression: str) -> str:
    """Evaluates a safe mathematical expression containing basic arithmetic."""
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        # Evaluate within safe isolated namespace
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


LOCAL_TOOLS = [get_current_time, calculate_expression]
