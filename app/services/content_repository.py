from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo.errors import ConfigurationError, PyMongoError

from app.db.mongo import MongoConnectionError, get_database


class ContentRepository:
    collection_name = "ai_contents"

    @property
    def collection(self):
        return get_database()[self.collection_name]

    async def create_content(self, data: dict[str, Any]) -> str:
        now = datetime.now(UTC)
        document = {
            **data,
            "prompt_versions": data.get("prompt_versions", []),
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = await self.collection.insert_one(document)
            return str(result.inserted_id)
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while creating content.") from exc

    async def get_content(self, user_id: str, content_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(content_id):
            return None

        try:
            document = await self.collection.find_one(
                {"_id": ObjectId(content_id), "user_id": user_id}
            )
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while reading content.") from exc

        if document is None:
            return None
        document["_id"] = str(document["_id"])
        document["content_id"] = document["_id"]
        return document

    async def replace_refined_prompt(
        self,
        user_id: str,
        content_id: str,
        new_refined_prompt: str,
        openai_cost: float,
        openai_cost_formatted: str,
        openai_usage: dict[str, Any],
    ) -> dict[str, Any] | None:
        document = await self.get_content(user_id=user_id, content_id=content_id)
        if document is None:
            return None

        now = datetime.now(UTC)
        previous_versions = document.get("prompt_versions")
        version_number = len(previous_versions) + 1 if isinstance(previous_versions, list) else 1

        old_version = {
            "version": version_number,
            "AI_refine_prompt": document.get("ai_refined_prompt", ""),
            "openAI_cost": document.get("last_openai_cost_formatted", "$0.00"),
            "openai_cost": document.get("last_openai_cost", 0.0),
            "openai_usage": document.get("last_openai_usage", {}),
            "created_at": document.get("updated_at") or document.get("created_at") or now,
        }

        try:
            await self.collection.update_one(
                {"_id": ObjectId(content_id), "user_id": user_id},
                {
                    "$set": {
                        "ai_refined_prompt": new_refined_prompt,
                        "last_openai_cost": openai_cost,
                        "last_openai_cost_formatted": openai_cost_formatted,
                        "last_openai_usage": openai_usage,
                        "updated_at": now,
                    },
                    "$inc": {"openai_cost_total": openai_cost},
                    "$push": {"prompt_versions": old_version},
                },
            )
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while updating content.") from exc

        return await self.get_content(user_id=user_id, content_id=content_id)
