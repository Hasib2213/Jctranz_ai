from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.core.redis import get_redis_settings
from app.models.ai_models import MediaGenerationRequest

logger = logging.getLogger(__name__)


async def task_generate_image(ctx: dict[str, Any], user_id: str, content_id: str) -> dict[str, Any]:
    from app.services.media_generation_service import MediaGenerationService

    logger.info("ARQ Worker: Starting image generation for user=%s, content_id=%s", user_id, content_id)
    service = MediaGenerationService()
    payload = MediaGenerationRequest(user_id=user_id, content_id=content_id)
    response = await service._execute_direct_image_generation(payload)
    return response.model_dump()


async def task_generate_video(ctx: dict[str, Any], user_id: str, content_id: str) -> dict[str, Any]:
    from app.services.media_generation_service import MediaGenerationService

    logger.info("ARQ Worker: Starting video generation for user=%s, content_id=%s", user_id, content_id)
    service = MediaGenerationService()
    payload = MediaGenerationRequest(user_id=user_id, content_id=content_id)
    response = await service._execute_direct_video_generation(payload)
    return response.model_dump()


async def task_generate_video_edit(ctx: dict[str, Any], user_id: str, content_id: str) -> dict[str, Any]:
    from app.services.media_generation_service import MediaGenerationService

    logger.info("ARQ Worker: Starting video edit for user=%s, content_id=%s", user_id, content_id)
    service = MediaGenerationService()
    payload = MediaGenerationRequest(user_id=user_id, content_id=content_id)
    response = await service._execute_direct_video_edit(payload)
    return response.model_dump()


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("ARQ Worker starting up...")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("ARQ Worker shutting down...")


settings = get_settings()


class WorkerSettings:
    functions = [
        task_generate_image,
        task_generate_video,
        task_generate_video_edit,
    ]
    redis_settings = get_redis_settings()
    max_jobs = settings.arq_max_jobs
    job_timeout = settings.arq_job_timeout
    on_startup = startup
    on_shutdown = shutdown


if __name__ == "__main__":
    from arq import run_worker

    run_worker(WorkerSettings)
