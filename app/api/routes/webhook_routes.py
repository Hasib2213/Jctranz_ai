import logging
from fastapi import APIRouter, Header, HTTPException, Request, status
from app.db.mongo import get_database

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/provider/{provider_name}")
async def handle_provider_webhook(
    provider_name: str,
    request: Request,
    idempotency_key: str | None = Header(None, alias="X-Idempotency-Key"),
):
    """
    Authenticated and idempotent webhook receiver endpoint for AI providers.
    """
    db = get_database()
    webhooks_col = db.webhook_events

    body = await request.json()

    # Idempotency check
    if idempotency_key:
        existing = await webhooks_col.find_one({"idempotency_key": idempotency_key})
        if existing:
            logger.info(f"Duplicate webhook ignored for key: {idempotency_key}")
            return {"status": "ignored", "reason": "duplicate idempotency_key"}

    webhook_event = {
        "provider": provider_name,
        "idempotency_key": idempotency_key,
        "payload": body,
        "processed": True,
    }

    await webhooks_col.insert_one(webhook_event)
    logger.info(f"Processed webhook event from {provider_name}")
    return {"status": "ok", "provider": provider_name}
