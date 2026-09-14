from datetime import datetime, timezone
import uuid
import logging
from app.db.mongo import get_database

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(self) -> None:
        self.db = get_database()
        self.audit_logs = self.db.generation_audit_logs

    async def log_generation(
        self,
        user_id: str,
        generation_type: str,
        provider: str,
        model: str,
        credits: float,
        provider_cost: float = 0.0,
        status: str = "completed",
        error_code: str | None = None,
        asset_id: str | None = None,
        duration_seconds: float | None = None,
        parameters: dict | None = None,
    ) -> str:
        generation_id = f"gen_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        record = {
            "generation_id": generation_id,
            "user_id": user_id,
            "type": generation_type,
            "provider": provider,
            "model": model,
            "parameters": parameters or {},
            "duration_seconds": duration_seconds,
            "credits": credits,
            "provider_cost": provider_cost,
            "status": status,
            "error_code": error_code,
            "asset_id": asset_id,
            "created_at": now,
        }

        await self.audit_logs.insert_one(record)
        logger.info(f"Logged generation audit record: {generation_id} for user {user_id} ({status})")
        return generation_id
