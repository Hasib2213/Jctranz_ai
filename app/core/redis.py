import logging
import time
from typing import Any

from arq.connections import ArqRedis, RedisSettings, create_pool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_redis_pool: ArqRedis | None = None
_last_failure_time: float = 0.0
_failure_cooldown_seconds: float = 30.0


def get_redis_settings() -> RedisSettings:
    settings = get_settings()
    parsed = RedisSettings.from_dsn(settings.redis_url)
    return RedisSettings(
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
        username=parsed.username,
        password=parsed.password,
        ssl=parsed.ssl,
        conn_retries=1,
        conn_timeout=1,
    )


async def get_arq_redis() -> ArqRedis | None:
    """
    Returns the cached ARQ Redis pool.
    If not yet connected, attempts to connect.
    If Redis is unavailable, cools down and returns None gracefully for instant fallback.
    """
    global _redis_pool, _last_failure_time
    if _redis_pool is not None:
        return _redis_pool

    settings = get_settings()
    if not settings.arq_enabled:
        return None

    now = time.time()
    if (now - _last_failure_time) < _failure_cooldown_seconds:
        return None

    try:
        redis_settings = get_redis_settings()
        _redis_pool = await create_pool(redis_settings)
        logger.info("Successfully connected to Redis ARQ pool at %s", settings.redis_url)
        return _redis_pool
    except Exception as exc:
        _last_failure_time = now
        logger.warning(
            "Could not connect to Redis at %s (%s). Falling back to direct execution.",
            settings.redis_url,
            exc,
        )
        return None



async def close_arq_redis() -> None:
    """
    Gracefully closes the ARQ Redis pool on shutdown.
    """
    global _redis_pool
    if _redis_pool is not None:
        try:
            await _redis_pool.close()
            logger.info("Closed Redis ARQ pool connection.")
        except Exception as exc:
            logger.warning("Error closing Redis ARQ pool: %s", exc)
        finally:
            _redis_pool = None
