from app.models.ai_models import (
    JobStatus,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from app.services.fal_service import FalVideoService
from app.services.job_repository import JobRepository
from app.services.openai_service import OpenAIScriptService


class AIWorkflowService:
    def __init__(self) -> None:
        self.jobs = JobRepository()
        self.openai = OpenAIScriptService()
        self.fal = FalVideoService()

    async def generate_script(self, payload: ScriptGenerationRequest) -> ScriptGenerationResponse:
        job_id = await self.jobs.create_job(
            {
                "status": JobStatus.PENDING,
                "product_name": payload.product_name,
                "product_description": payload.product_description,
                "creator_prompt": payload.creator_prompt,
            }
        )

        try:
            script = await self.openai.generate_promotional_script(payload)
            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.SCRIPT_GENERATED,
                    "promotional_script": script,
                },
            )
            return ScriptGenerationResponse(job_id=job_id, promotional_script=script)
        except Exception as exc:
            await self.jobs.mark_failed(job_id, str(exc))
            raise

    async def generate_video(self, payload: VideoGenerationRequest) -> VideoGenerationResponse:
        job_id = payload.job_id
        if job_id is None:
            job_id = await self.jobs.create_job(
                {
                    "status": JobStatus.VIDEO_GENERATING,
                    "product_name": payload.product_name,
                    "product_image_url": str(payload.product_image_url),
                    "promotional_script": payload.approved_script,
                }
            )
        else:
            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.VIDEO_GENERATING,
                    "product_name": payload.product_name,
                    "product_image_url": str(payload.product_image_url),
                    "promotional_script": payload.approved_script,
                },
            )

        try:
            video_url, provider_response, final_prompt = await self.fal.generate_video(payload)
            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.VIDEO_COMPLETED,
                    "final_video_prompt": final_prompt,
                    "video_url": video_url,
                    "provider_response": provider_response,
                },
            )
            return VideoGenerationResponse(
                job_id=job_id,
                video_url=video_url,
                provider_response=provider_response,
            )
        except Exception as exc:
            await self.jobs.mark_failed(job_id, str(exc))
            raise

    async def get_job(self, job_id: str):
        return await self.jobs.get_job(job_id)
