from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAITokenCost(BaseModel):
    input_token: int = Field(default=0, description="Input prompt tokens consumed")
    output_token: int = Field(default=0, description="Output completion tokens generated")
    total_token: int | None = Field(default=None, description="Total tokens consumed")

    model_config = ConfigDict(populate_by_name=True)


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120, description="Unique identifier of the user")
    user_message: str = Field(..., min_length=1, max_length=10000, description="User's input prompt or question")


class ChatResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user")
    AI_response: str = Field(..., description="Generated AI response message")
    openAI_token_cost: OpenAITokenCost = Field(..., description="OpenAI token usage details")

    model_config = ConfigDict(populate_by_name=True)


class ChatMessageItem(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Role of the message sender")
    message: str = Field(..., description="Message text content")
    created_at: datetime | None = Field(default=None, description="Timestamp when message was stored")

    model_config = ConfigDict(populate_by_name=True)


class ChatHistoryResponse(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user")
    total_messages: int = Field(default=0, description="Total number of messages in history")
    messages: list[ChatMessageItem] = Field(default_factory=list, description="List of messages in chronological order")

    model_config = ConfigDict(populate_by_name=True)
