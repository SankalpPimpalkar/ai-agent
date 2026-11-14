import json
from typing import Dict, List, Optional, Tuple


class LoopBreakers:
    """
    Detects and breaks repetitive loops or cascading errors during agent execution.
    1. Consecutive errors: triggers at >= max_consecutive_errors (default 4).
    2. Repeated calls: triggers at >= max_repeated_calls (default 3 identical calls with same result).
    """

    def __init__(
        self,
        max_consecutive_errors: int = 4,
        max_repeated_calls: int = 3,
    ):
        self.max_consecutive_errors = max_consecutive_errors
        self.max_repeated_calls = max_repeated_calls
        self.consecutive_errors: int = 0
        self.seen_calls: Dict[Tuple[str, str, str], int] = {}
        self.triggered_breakers: List[str] = []

    def check(
        self,
        name: str,
        args: dict,
        result: str,
        step: int,
        logs: Optional[List[str]] = None,
    ) -> str:
        """
        Evaluates a tool call outcome against loop breaker rules.
        Returns the (possibly annotated) result string.
        """
        # 1. Consecutive errors check
        if result.startswith("Error"):
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        if self.consecutive_errors >= self.max_consecutive_errors:
            result += "\nToo many errors in a row. Stop and tell the user what is failing."
            if "errors" not in self.triggered_breakers:
                self.triggered_breakers.append("errors")
            if logs is not None:
                logs.append(f"[Step {step}] Breaker triggered: 4 consecutive errors")

        # 2. Repeated identical calls check
        key = (name, json.dumps(args, sort_keys=True, default=str), result)
        self.seen_calls[key] = self.seen_calls.get(key, 0) + 1
        if self.seen_calls[key] >= self.max_repeated_calls:
            result += "\nYou have repeated this exact call 3 times with the same result. Stop and tell the user what is failing."
            if "repeat" not in self.triggered_breakers:
                self.triggered_breakers.append("repeat")
            if logs is not None:
                logs.append(
                    f"[Step {step}] Breaker triggered: call repeated 3 times with same result"
                )

        return result

    def reset(self) -> None:
        """Resets all breaker counters and tracked history."""
        self.consecutive_errors = 0
        self.seen_calls.clear()
        self.triggered_breakers.clear()
