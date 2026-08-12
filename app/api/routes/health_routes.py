from fastapi import APIRouter

from app.db.mongo import MongoConnectionError, check_mongo_connection

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/health/db")
async def database_health_check():
    try:
        return await check_mongo_connection()
    except MongoConnectionError as exc:
        return {"status": "error", "detail": str(exc)}
