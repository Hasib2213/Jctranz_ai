from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from pymongo.errors import ConfigurationError, PyMongoError

from app.db.mongo import MongoConnectionError, get_database

logger = logging.getLogger(__name__)


class ChatRepository:
    collection_name = "chat_messages"

    def __init__(self) -> None:
        self._index_created = False

    @property
    def collection(self):
        return get_database()[self.collection_name]

    async def _ensure_indexes(self) -> None:
        if self._index_created:
            return
        try:
            await self.collection.create_index([("user_id", 1), ("created_at", 1)])
            self._index_created = True
        except Exception as exc:
            logger.warning("Could not create index on chat_messages: %s", exc)

    async def save_chat_turn(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_message: str,
        tokens: dict[str, Any] | None = None,
    ) -> None:
        await self._ensure_indexes()
        now = datetime.now(UTC)

        user_doc = {
            "user_id": user_id,
            "role": "user",
            "message": user_message,
            "created_at": now,
        }
        assistant_doc = {
            "user_id": user_id,
            "role": "assistant",
            "message": assistant_message,
            "tokens": tokens or {},
            "created_at": now,
        }

        try:
            await self.collection.insert_many([user_doc, assistant_doc])
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while saving chat messages.") from exc

    async def get_recent_context(self, user_id: str, limit: int = 10) -> list[dict[str, str]]:
        """
        Retrieves the latest `limit` messages in chronological order formatted for OpenAI.
        """
        await self._ensure_indexes()
        try:
            cursor = (
                self.collection.find({"user_id": user_id})
                .sort("created_at", -1)
                .limit(limit)
            )
            raw_messages = await cursor.to_list(length=limit)
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while reading recent chat context.") from exc

        # Reverse so older messages come before newer messages
        raw_messages.reverse()
        return [
            {
                "role": doc.get("role", "user"),
                "content": doc.get("message", ""),
            }
            for doc in raw_messages
            if doc.get("role") in {"user", "assistant"} and doc.get("message")
        ]

    async def get_user_chat_history(self, user_id: str, limit: int = 300) -> list[dict[str, Any]]:
        """
        Retrieves full chat history for a user sorted chronologically.
        """
        await self._ensure_indexes()
        try:
            cursor = (
                self.collection.find({"user_id": user_id})
                .sort("created_at", 1)
                .limit(limit)
            )
            docs = await cursor.to_list(length=limit)
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while reading chat history.") from exc

        return [
            {
                "role": doc.get("role", "user"),
                "message": doc.get("message", ""),
                "created_at": doc.get("created_at"),
            }
            for doc in docs
        ]
