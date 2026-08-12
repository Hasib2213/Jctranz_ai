from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConfigurationError, PyMongoError

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None


class MongoConnectionError(RuntimeError):
    pass


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return get_mongo_client()[settings.mongodb_db_name]


async def check_mongo_connection() -> dict[str, str]:
    settings = get_settings()
    try:
        await get_database().command("ping")
        return {
            "status": "ok",
            "database": settings.mongodb_db_name,
        }
    except (ConfigurationError, PyMongoError, OSError) as exc:
        raise MongoConnectionError(
            "MongoDB connection failed. Check MONGODB_URI, Atlas cluster host, "
            "database user/password, and Network Access IP allowlist. "
            f"Original error: {exc}"
        ) from exc


async def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
