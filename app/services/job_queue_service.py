import asyncio
import logging
from typing import Any, Callable, Coroutine
from app.services.job_repository import JobRepository
from app.models.ai_models import JobStatus

logger = logging.getLogger(__name__)


class JobQueueService:
    def __init__(self) -> None:
        self.jobs = JobRepository()

    async def enqueue_job(
        self, job_id: str, async_task_fn: Callable[[], Coroutine[Any, Any, None]]
    ) -> str:
        """
        Enqueues an async job in background task queue.
        Updates job status to 'queued' and launches execution in background.
        """
        await self.jobs.update_job(job_id, {"status": JobStatus.PENDING})

        async def worker_wrapper():
            try:
                await self.jobs.update_job(job_id, {"status": JobStatus.VIDEO_GENERATING})
                await async_task_fn()
            except Exception as exc:
                logger.error(f"Background Job {job_id} execution failed: {exc}")
                await self.jobs.mark_failed(job_id, str(exc))

        # Dispatch background worker task
        asyncio.create_task(worker_wrapper())
        logger.info(f"Enqueued async job {job_id} successfully.")
        return job_id
