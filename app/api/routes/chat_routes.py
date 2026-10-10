from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.chat_models import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse, include_in_schema=False)
async def create_chat_message(payload: ChatRequest) -> ChatResponse:
    """
    Send a message to RENE, the official FMFO AI Assistant.
    Provides answers to platform queries, creator support, video tutorial walkthroughs, and creative suggestions (captions, content ideas)
    with conversational context preservation and token usage metrics.
    """
    service = ChatService()
    return await service.process_chat_message(payload)


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    user_id: str = Query(..., min_length=1, description="User ID to fetch chat history for")
) -> ChatHistoryResponse:
    """
    Retrieve full chronological chat history for a specific user.
    """
    service = ChatService()
    return await service.get_chat_history(user_id=user_id)
