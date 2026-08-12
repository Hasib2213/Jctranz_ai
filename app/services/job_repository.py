from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from pymongo.errors import ConfigurationError, PyMongoError

from app.db.mongo import MongoConnectionError, get_database
from app.models.ai_models import JobStatus


class JobRepository:
    collection_name = "ai_workflow_jobs"

    @property
    def collection(self):
        return get_database()[self.collection_name]

    async def create_job(self, data: dict[str, Any]) -> str:
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
            raise MongoConnectionError(
                "MongoDB connection failed. Your MONGODB_URI is not connecting. "
                "Copy the exact URI from MongoDB Atlas -> Connect -> Drivers."
            ) from exc

    async def update_job(self, job_id: str, data: dict[str, Any]) -> None:
        if not ObjectId.is_valid(job_id):
            return

        try:
            await self.collection.update_one(
                {"_id": ObjectId(job_id)},
                {
                    "$set": {
                        **data,
                        "updated_at": datetime.now(UTC),
                    }
                },
            )
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while updating job.") from exc

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        if not ObjectId.is_valid(job_id):
            return None

        try:
            document = await self.collection.find_one({"_id": ObjectId(job_id)})
        except (ConfigurationError, PyMongoError, OSError) as exc:
            raise MongoConnectionError("MongoDB connection failed while reading job.") from exc
        if document is None:
            return None
        document["_id"] = str(document["_id"])
        return document

    async def mark_failed(self, job_id: str, error_message: str) -> None:
        await self.update_job(
            job_id,
            {
                "status": JobStatus.FAILED,
                "error_message": error_message,
            },
        )
