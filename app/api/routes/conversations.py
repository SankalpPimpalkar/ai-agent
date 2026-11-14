from typing import List
from fastapi import APIRouter, HTTPException, Query, status
from app.agent.memory import (
    delete_conversation_session,
    format_message_for_api,
    get_conversation_session,
    list_conversation_sessions,
    load_session_history,
)
from app.schemas.chat import ConversationDetailResponse, ConversationSessionItem

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=List[ConversationSessionItem])
async def list_conversations(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of conversations to return",
    ),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    Returns the list of active conversations with their session IDs,
    auto-updated topic summaries, and last message timestamps.
    """
    records = await list_conversation_sessions(limit=limit, offset=offset)
    return [
        ConversationSessionItem(
            session_id=r.session_id,
            summary=r.summary,
            last_message_at=r.last_message_at,
            created_at=r.created_at,
            message_count=r.message_count,
        )
        for r in records
    ]


@router.get("/{session_id}", response_model=ConversationDetailResponse)
async def get_conversation(session_id: str):
    """
    Returns conversation metadata along with its full message history.
    """
    record = await get_conversation_session(session_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found",
        )

    raw_messages = await load_session_history(session_id)
    formatted_messages = [format_message_for_api(m) for m in raw_messages]

    return ConversationDetailResponse(
        session_id=record.session_id,
        summary=record.summary,
        last_message_at=record.last_message_at,
        created_at=record.created_at,
        message_count=record.message_count,
        messages=formatted_messages,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(session_id: str):
    """
    Deletes a conversation session from the database.
    """
    deleted = await delete_conversation_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation session '{session_id}' not found",
        )
