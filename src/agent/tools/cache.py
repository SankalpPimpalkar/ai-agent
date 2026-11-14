import json
from typing import Any, Dict
from src.agent.tools.base import BaseTool
from src.agent.tools.state import STATE


class CacheTool(BaseTool):
    """
    Manages temporary task cache and unified state across tools.
    Actions: set, get, list, delete, clear.
    """

    @property
    def name(self) -> str:
        return "cache_tool"

    @property
    def description(self) -> str:
        return """
Manage temporary task cache and unified state across tools.
Actions: set, get, list, delete, clear.
""".strip()

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "list", "delete", "clear"],
                    "description": "Cache operation to perform",
                },
                "key": {
                    "type": "string",
                    "description": "Key identifier for get/set/delete",
                },
                "value": {
                    "description": "Value to store for set action",
                },
            },
            "required": ["action"],
        }

    def run(self, action: str, key: str = "", value: Any = None, **kwargs) -> str:
        try:
            action = (action or "").lower().strip()
            if action in ("list", "all"):
                STATE.memory_listed = True
                data = getattr(STATE, "cache", {})
                if not data:
                    return "Cache is empty"
                lines = [
                    f"{k}: {json.dumps(v) if not isinstance(v, str) else v}"
                    for k, v in data.items()
                ]
                return "\n".join(lines)

            if action == "clear":
                if hasattr(STATE, "cache"):
                    STATE.cache.clear()
                return "Cache cleared"

            if not key:
                return f"Error: 'key' is required for action '{action}'"

            if not hasattr(STATE, "cache"):
                STATE.cache = {}

            if action == "get":
                if key not in STATE.cache:
                    return f"Key '{key}' not found in cache"
                val = STATE.cache[key]
                return (
                    str(val) if isinstance(val, (int, float, str)) else json.dumps(val)
                )

            if action == "set":
                if value is None:
                    return "Error: 'value' is required for action 'set'"
                parsed_val = value
                if isinstance(value, str):
                    try:
                        parsed_val = json.loads(value)
                    except Exception:
                        parsed_val = value
                STATE.cache[key] = parsed_val
                return f"Cached '{key}' = {parsed_val}"

            if action == "delete":
                if key in STATE.cache:
                    del STATE.cache[key]
                    return f"Deleted key '{key}' from cache"
                return f"Key '{key}' was not in cache"

            return (
                f"Error: unknown action '{action}'. Use set, get, list, delete, or clear"
            )
        except Exception as e:
            return f"Error: {e}"


cache_instance = CacheTool()


def cache_tool(action: str, key: str = "", value: Any = None) -> str:
    return cache_instance.run(action=action, key=key, value=value)


cache_tool_spec = cache_instance.to_spec()
