from datetime import datetime, timezone
import uuid
import logging
from app.db.mongo import get_database
from app.core.errors import InsufficientCreditsException

logger = logging.getLogger(__name__)


class CreditService:
    def __init__(self) -> None:
        self.db = get_database()
        self.wallets = self.db.wallets
        self.ledger = self.db.credit_ledger

    async def get_wallet(self, user_id: str) -> dict:
        wallet = await self.wallets.find_one({"user_id": user_id})
        if not wallet:
            # Grant initial default credits (e.g. 100.0) for new users
            new_wallet = {
                "user_id": user_id,
                "balance_credits": 100.0,
                "reserved_credits": 0.0,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
            await self.wallets.insert_one(new_wallet)
            return new_wallet
        return wallet

    async def reserve_credits(self, user_id: str, job_id: str, required_credits: float) -> str:
        """
        Reserves credits before starting an AI job to prevent double spending.
        """
        wallet = await self.get_wallet(user_id)
        available_credits = wallet.get("balance_credits", 0.0) - wallet.get("reserved_credits", 0.0)

        if available_credits < required_credits:
            logger.warning(f"User {user_id} insufficient credits. Available: {available_credits}, Required: {required_credits}")
            raise InsufficientCreditsException(
                f"Insufficient credits. You have {available_credits} available, but {required_credits} required."
            )

        transaction_id = f"tx_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Atomic reserve update
        result = await self.wallets.update_one(
            {"user_id": user_id},
            {
                "$inc": {"reserved_credits": required_credits},
                "$set": {"updated_at": now},
            },
        )

        # Create Ledger Record (Status: reserved)
        await self.ledger.insert_one(
            {
                "transaction_id": transaction_id,
                "user_id": user_id,
                "job_id": job_id,
                "credits": required_credits,
                "status": "reserved",
                "created_at": now,
                "updated_at": now,
            }
        )

        logger.info(f"Reserved {required_credits} credits for user {user_id}, Tx: {transaction_id}")
        return transaction_id

    async def commit_credits(
        self, user_id: str, transaction_id: str, generation_id: str, credits: float
    ) -> dict:
        """
        Finalizes credit consumption after successful AI generation.
        Deducts balance_credits and releases reserved_credits.
        """
        now = datetime.now(timezone.utc)
        await self.wallets.update_one(
            {"user_id": user_id},
            {
                "$inc": {
                    "balance_credits": -credits,
                    "reserved_credits": -credits,
                },
                "$set": {"updated_at": now},
            },
        )

        await self.ledger.update_one(
            {"transaction_id": transaction_id},
            {
                "$set": {
                    "status": "consumed",
                    "generation_id": generation_id,
                    "updated_at": now,
                }
            },
        )
        logger.info(f"Committed {credits} credits for user {user_id}, Tx: {transaction_id}")
        return {"status": "consumed", "transaction_id": transaction_id}

    async def refund_credits(
        self, user_id: str, transaction_id: str, generation_id: str, credits: float, reason: str = ""
    ) -> dict:
        """
        Releases reserved credits back to user on AI generation failure.
        """
        now = datetime.now(timezone.utc)
        await self.wallets.update_one(
            {"user_id": user_id},
            {
                "$inc": {"reserved_credits": -credits},
                "$set": {"updated_at": now},
            },
        )

        await self.ledger.update_one(
            {"transaction_id": transaction_id},
            {
                "$set": {
                    "status": "refunded",
                    "generation_id": generation_id,
                    "refund_reason": reason,
                    "updated_at": now,
                }
            },
        )
        logger.info(f"Refunded {credits} credits to user {user_id}, Tx: {transaction_id}, Reason: {reason}")
        return {"status": "refunded", "transaction_id": transaction_id}
