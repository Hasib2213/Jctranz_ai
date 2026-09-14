from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo.errors import ConfigurationError, PyMongoError

from app.db.mongo import MongoConnectionError, get_database


class GenerationRepository:
    collection_name = "ai_generations"

    @property
    def collection(self):
        return get_database()[self.collection_name]

    async def create_generation(self, data: dict[str, Any]) -> str:
        now = datetime.now(UTC)
        document = {
            **data,
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = await self.collection.insert_one(document)
            return str(result.inserted_id)
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while creating generation.") from exc

    async def get_generation(
        self, user_id: str, generated_content_id: str
    ) -> dict[str, Any] | None:
        if not ObjectId.is_valid(generated_content_id):
            return None

        try:
            document = await self.collection.find_one(
                {"_id": ObjectId(generated_content_id), "user_id": user_id}
            )
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while reading generation.") from exc

        if document is None:
            return None
        document["_id"] = str(document["_id"])
        document["generated_content_id"] = document["_id"]
        return document
