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
        openai_usage: dict[str, Any],
        agentic_plan: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        document = await self.get_content(user_id=user_id, content_id=content_id)
        if document is None:
            return None

        now = datetime.now(UTC)
        previous_versions = document.get("prompt_versions")
        version_number = len(previous_versions) + 1 if isinstance(previous_versions, list) else 1

        old_tokens = (
            document.get("last_openai_usage")
            or document.get("openai_tokens")
            or {}
        )
        if not isinstance(old_tokens, dict):
            old_tokens = {}

        old_version = {
            "version": version_number,
            "AI_refine_prompt": document.get("ai_refined_prompt", ""),
            "openai_tokens": {
                "input_tokens": int(old_tokens.get("input_tokens") or 0),
                "output_tokens": int(old_tokens.get("output_tokens") or 0),
                "total_tokens": int(old_tokens.get("total_tokens") or 0),
            },
            "created_at": document.get("updated_at") or document.get("created_at") or now,
        }

        clean_usage = {
            "input_tokens": int(openai_usage.get("input_tokens") or 0),
            "output_tokens": int(openai_usage.get("output_tokens") or 0),
            "total_tokens": int(openai_usage.get("total_tokens") or 0),
        }

        try:
            update_set: dict[str, Any] = {
                "ai_refined_prompt": new_refined_prompt,
                "last_openai_usage": clean_usage,
                "openai_tokens": clean_usage,
                "updated_at": now,
            }
            if agentic_plan is not None:
                update_set["agentic_plan"] = agentic_plan
            if settings is not None:
                update_set["settings"] = settings

            existing_total = document.get("openai_tokens_total")
            update_doc: dict[str, Any] = {
                "$set": update_set,
                "$push": {"prompt_versions": old_version},
            }

            if not isinstance(existing_total, dict):
                update_set["openai_tokens_total"] = {
                    "input_tokens": int(old_tokens.get("input_tokens") or 0) + clean_usage["input_tokens"],
                    "output_tokens": int(old_tokens.get("output_tokens") or 0) + clean_usage["output_tokens"],
                    "total_tokens": int(old_tokens.get("total_tokens") or 0) + clean_usage["total_tokens"],
                }
            else:
                update_doc["$inc"] = {
                    "openai_tokens_total.input_tokens": clean_usage["input_tokens"],
                    "openai_tokens_total.output_tokens": clean_usage["output_tokens"],
                    "openai_tokens_total.total_tokens": clean_usage["total_tokens"],
                }

            await self.collection.update_one(
                {"_id": ObjectId(content_id), "user_id": user_id},
                update_doc,
            )
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while updating content.") from exc

        return await self.get_content(user_id=user_id, content_id=content_id)
