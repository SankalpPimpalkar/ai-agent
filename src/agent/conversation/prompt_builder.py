from typing import Any, List, Optional
from ollama import Message
from src.agent.settings import SETTINGS


def _role(m: Any) -> str:
    return m["role"] if isinstance(m, dict) else m.role


class PromptBuilder:
    """
    Constructs message payloads for model turns with system prompts,
    tool instructions, memory indexes, and history sanitization.
    """

    def __init__(self, settings=None):
        self.settings = settings or SETTINGS

    def build_messages(
        self,
        chat_history: Optional[List[Message | dict]] = None,
        tool_use: bool = True,
        system_prompt: Optional[str] = None,
        tool_instructions: Optional[str] = None,
        memory_instructions: Optional[str] = None,
        memory_index_text: Optional[str] = None,
    ) -> list:
        history = chat_history if chat_history is not None else []

        if system_prompt is not None:
            base_system_prompt = system_prompt
        elif history and _role(history[0]) == "system":
            first_msg = history[0]
            base_system_prompt = (
                first_msg.content
                if hasattr(first_msg, "content")
                else first_msg.get("content", self.settings.SYSTEM_PROMPT)
            )
        else:
            base_system_prompt = self.settings.SYSTEM_PROMPT

        tool_inst = (
            tool_instructions
            if tool_instructions is not None
            else self.settings.TOOL_INSTRUCTIONS
        )
        mem_inst = (
            memory_instructions
            if memory_instructions is not None
            else self.settings.MEMORY_INSTRUCTIONS
        )

        mem_idx = memory_index_text
        if mem_idx is None:
            try:
                from src.agent.tools.memory.store import memory_index
                mem_idx = memory_index()
            except ImportError:
                mem_idx = ""

        if tool_use:
            system_text = "\n\n".join([
                base_system_prompt.strip(),
                tool_inst.strip(),
                mem_inst.strip(),
                f"Your memory files:\n{mem_idx}",
            ])
        else:
            system_text = (
                f"{base_system_prompt.strip()}\n\n"
                "You have no tools in this mode. Do not claim to use tools, "
                "run code, or save or read memory."
            )

        raw_rest = (
            history[1:] if (history and _role(history[0]) == "system") else history[:]
        )
        rest = []
        for m in raw_rest:
            if not tool_use:
                d = m if isinstance(m, dict) else m.model_dump()
                if d.get("role") == "tool":
                    continue
                if d.get("tool_calls"):
                    if not d.get("content"):
                        continue
                    d = {**d, "tool_calls": None}
                rest.append(d)
            else:
                if (
                    hasattr(m, "role")
                    and m.role == "assistant"
                    and getattr(m, "thinking", None)
                ):
                    rest.append(
                        Message(
                            role=m.role,
                            content=m.content,
                            tool_calls=m.tool_calls,
                        )
                    )
                elif (
                    isinstance(m, dict)
                    and m.get("role") == "assistant"
                    and m.get("thinking")
                ):
                    rest.append({k: v for k, v in m.items() if k != "thinking"})
                else:
                    rest.append(m)

        return [{"role": "system", "content": system_text}, *rest]
