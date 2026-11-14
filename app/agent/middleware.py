from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.config import settings


def prune_old_tool_outputs_middleware(
    messages: List[BaseMessage],
    min_length_to_prune: int = 40,
) -> List[BaseMessage]:
    """
    Middleware that converts older tool outputs from previous turns into short placeholders.
    Preserves tool_call_id and name so that message schemas remain strictly valid,
    while freeing up context window tokens from past tool executions.
    """
    result: List[BaseMessage] = []
    total = len(messages)

    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            # A ToolMessage is considered 'old' if there is an AIMessage later in history,
            # indicating that the turn has completed and the model already processed its output.
            is_old = any(isinstance(later, AIMessage) for later in messages[i + 1 :])
            if is_old and len(msg.content) > min_length_to_prune:
                tool_name = msg.name or "tool"
                placeholder = f"[Tool '{tool_name}' output ({len(msg.content)} chars) archived]"
                result.append(
                    ToolMessage(
                        content=placeholder,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name,
                    )
                )
                continue
        result.append(msg)

    return result


async def summarize_context_middleware(
    messages: List[BaseMessage],
    llm: Optional[BaseChatModel] = None,
    max_context_chars: Optional[int] = None,
    max_context_messages: Optional[int] = None,
    keep_recent: Optional[int] = None,
) -> List[BaseMessage]:
    """
    Conversation compression middleware.
    When the total conversation size (in characters or message count) overflows
    the configured thresholds, older conversation turns are summarized and compressed
    into a SystemMessage, preserving the initial system prompt and recent turns.
    """
    max_chars = max_context_chars or settings.MAX_CONTEXT_CHARACTERS
    max_msgs = max_context_messages or settings.MAX_CONTEXT_MESSAGES
    recent_k = keep_recent or settings.SUMMARY_KEEP_RECENT

    total_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    if total_chars <= max_chars and len(messages) <= max_msgs:
        return messages

    # Keep initial SystemMessage intact if present
    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    rest = messages[1:] if system_msg else messages

    if len(rest) <= recent_k:
        return messages

    # Find a clean boundary for recent messages (do not orphan a ToolMessage from its AIMessage)
    split_idx = len(rest) - recent_k
    while split_idx > 0 and isinstance(rest[split_idx], ToolMessage):
        split_idx -= 1
    if split_idx <= 0:
        return messages

    to_summarize = rest[:split_idx]
    recent_messages = rest[split_idx:]

    summary_text = ""
    if llm is not None:
        try:
            transcript_lines = []
            for m in to_summarize:
                role = (
                    "User"
                    if isinstance(m, HumanMessage)
                    else "Assistant"
                    if isinstance(m, AIMessage)
                    else "Tool"
                    if isinstance(m, ToolMessage)
                    else "System"
                )
                transcript_lines.append(f"{role}: {getattr(m, 'content', '')}")
            transcript = "\n".join(transcript_lines)

            summary_prompt = [
                SystemMessage(
                    content=(
                        "You are an expert conversation summarizer. Create a concise summary of the "
                        "dialogue below. Preserve key facts, decisions, user preferences, and tool results."
                    )
                ),
                HumanMessage(
                    content=f"Please summarize this prior conversation concisely:\n\n{transcript}"
                ),
            ]
            response = await llm.ainvoke(summary_prompt)
            summary_text = str(getattr(response, "content", "")).strip()
        except Exception:
            summary_text = ""

    if not summary_text:
        # Fallback structured summary extraction
        extracted = []
        for m in to_summarize:
            if isinstance(m, HumanMessage) and m.content:
                extracted.append(f"- User requested: {m.content[:100]}")
            elif isinstance(m, AIMessage) and m.content:
                extracted.append(f"- Assistant responded: {m.content[:100]}")
        summary_text = "\n".join(extracted) if extracted else "Previous context archived."

    summary_msg = SystemMessage(
        content=f"Summary of previous conversation:\n{summary_text}"
    )

    new_messages: List[BaseMessage] = []
    if system_msg:
        new_messages.append(system_msg)
    new_messages.append(summary_msg)
    new_messages.extend(recent_messages)

    return new_messages
