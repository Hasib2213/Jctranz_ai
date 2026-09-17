from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.chat_models import (
    ChatHistoryResponse,
    ChatMessageItem,
    ChatRequest,
    ChatResponse,
    OpenAITokenCost,
)
from app.prompts.chat_prompt import get_fmfo_system_prompt
from app.services.chat_repository import ChatRepository
from app.services.cost_service import CostService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chat_repo = ChatRepository()
        self.costs = CostService()
        self.model = self.settings.openai_prompt_model or "gpt-4o-mini"
        self.api_key = self.settings.openai_api_key

    def _get_client(self) -> AsyncOpenAI:
        if not self.api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY is not configured in environment settings.",
            )
        return AsyncOpenAI(api_key=self.api_key)

    async def process_chat_message(self, payload: ChatRequest) -> ChatResponse:
        user_id = payload.user_id.strip()
        user_message = payload.user_message.strip()

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id cannot be empty.")
        if not user_message:
            raise HTTPException(status_code=400, detail="user_message cannot be empty.")

        # 1. Fetch recent conversation context (sliding window of last 10 messages)
        recent_context = await self.chat_repo.get_recent_context(user_id=user_id, limit=10)

        # 2. Build messages array with system prompt + history + current message
        messages: list[dict[str, str]] = [
            {"role": "system", "content": get_fmfo_system_prompt()}
        ]
        messages.extend(recent_context)
        messages.append({"role": "user", "content": user_message})

        # 3. Call OpenAI Chat Completions API
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
        except Exception as exc:
            logger.error("OpenAI chat completion failed for user_id=%s: %s", user_id, exc)
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI service error: {str(exc)}",
            ) from exc

        # 4. Extract generated response text
        choices = getattr(response, "choices", [])
        if not choices:
            raise HTTPException(
                status_code=502,
                detail="OpenAI returned an empty completion response.",
            )
        ai_response = (choices[0].message.content or "").strip()

        # 5. Extract token counts
        usage_info = self.costs.extract_openai_usage(response)
        input_tokens = usage_info["input_tokens"]
        output_tokens = usage_info["output_tokens"]
        total_tokens = usage_info["total_tokens"]

        token_cost = OpenAITokenCost(
            input_token=input_tokens,
            output_token=output_tokens,
            total_token=total_tokens,
        )

        # 6. Save both user message and AI response into MongoDB
        try:
            await self.chat_repo.save_chat_turn(
                user_id=user_id,
                user_message=user_message,
                assistant_message=ai_response,
                tokens=token_cost.model_dump(),
            )
        except Exception as exc:
            logger.error("Failed to persist chat messages to MongoDB for user_id=%s: %s", user_id, exc)
            # Log error but return response so user experience is not disrupted

        return ChatResponse(
            user_id=user_id,
            AI_response=ai_response,
            openAI_token_cost=token_cost,
        )

    async def get_chat_history(self, user_id: str) -> ChatHistoryResponse:
        cleaned_user_id = user_id.strip()
        if not cleaned_user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required.")

        history_docs = await self.chat_repo.get_user_chat_history(user_id=cleaned_user_id)
        messages = [
            ChatMessageItem(
                role=doc.get("role", "user"),
                message=doc.get("message", ""),
                created_at=doc.get("created_at"),
            )
            for doc in history_docs
        ]

        return ChatHistoryResponse(
            user_id=cleaned_user_id,
            total_messages=len(messages),
            messages=messages,
        )
