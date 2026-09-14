import logging
from typing import Any
from app.services.cloudinary_service import CloudinaryService
from app.services.credit_service import CreditService
from app.services.job_queue_service import JobQueueService
from app.services.job_repository import JobRepository
from app.services.monitoring_service import MonitoringService
from app.services.routing_service import RoutingService
from app.models.ai_models import (
    JobStatus,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    VideoGenerationContext,
    VideoGenerationRequest,
    VideoGenerationResponse,
)

logger = logging.getLogger(__name__)


class FMFOAIOrchestrator:
    def __init__(self) -> None:
        self.jobs = JobRepository()
        self.credits = CreditService()
        self.routing = RoutingService()
        self.storage = CloudinaryService()
        self.monitoring = MonitoringService()
        self.queue = JobQueueService()

    async def execute_script_workflow(
        self, payload: ScriptGenerationRequest, product_image_data_urls: list[str] | None = None
    ) -> ScriptGenerationResponse:
        """
        Orchestrator Flow for Script Generation:
        1. Validate & Create Job
        2. Reserve Credits
        3. Route to OpenAI Adapter
        4. Commit Credits
        5. Log Audit Record
        6. Return Normalized Result
        """
        # Step 1: Create Job
        job_id = await self.jobs.create_job(
            {
                "status": JobStatus.PENDING,
                "user_id": payload.user_id,
                "product_name": payload.product_name,
                "product_description": payload.product_description,
                "product_images": payload.product_images,
                "product_image_data_urls": product_image_data_urls,
                "time_seconds": payload.time_seconds,
            }
        )

        # Step 2: Reserve Credits (e.g. 5.0 credits for script generation)
        required_credits = 5.0
        tx_id = await self.credits.reserve_credits(
            user_id=payload.user_id, job_id=job_id, required_credits=required_credits
        )

        try:
            # Step 3: Route to Script Adapter
            adapter_res = await self.routing.execute_with_routing_and_failover(
                capability="script",
                payload={"task_type": "generate_script", "request_payload": payload},
            )

            script_data = adapter_res.get("result", {})
            scenes = script_data.get("scenes", [])

            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.SCRIPT_GENERATED,
                    "promotional_script": scenes,
                },
            )

            # Step 4: Commit Credits
            await self.credits.commit_credits(
                user_id=payload.user_id,
                transaction_id=tx_id,
                generation_id=job_id,
                credits=required_credits,
            )

            # Step 5: Log Monitoring Audit
            gen_id = await self.monitoring.log_generation(
                user_id=payload.user_id,
                generation_type="script",
                provider=adapter_res.get("provider", "openai"),
                model="gpt-4o-mini",
                credits=required_credits,
                status="completed",
                asset_id=job_id,
            )

            return ScriptGenerationResponse(
                job_id=job_id, promotional_script=scenes, status=JobStatus.SCRIPT_GENERATED
            )
        except Exception as exc:
            # Failure -> Refund Credits
            await self.credits.refund_credits(
                user_id=payload.user_id,
                transaction_id=tx_id,
                generation_id=job_id,
                credits=required_credits,
                reason=str(exc),
            )
            await self.jobs.mark_failed(job_id, str(exc))
            raise

    async def execute_video_workflow(
        self, payload: VideoGenerationRequest
    ) -> VideoGenerationResponse:
        """
        Orchestrator Flow for Video Generation:
        1. Validate & Fetch Job Context
        2. Reserve Credits
        3. Route to Fal.ai Video Adapter
        4. Upload Generated Asset to Cloudinary CDN (Storage)
        5. Commit Credits
        6. Log Audit Record
        7. Return Normalized Result
        """
        job = await self.jobs.get_job(payload.job_id)
        if not job or job.get("user_id") != payload.user_id:
            raise ValueError("Job not found or unauthorized.")

        product_name = job.get("product_name")
        approved_script = job.get("approved_script_text") or str(job.get("promotional_script"))

        resolved_payload = VideoGenerationContext(
            product_name=product_name,
            product_image_url=job.get("product_image_url"),
            approved_script=approved_script,
            time_seconds=job.get("time_seconds") or 15,
        )

        job_id = payload.job_id
        required_credits = 20.0  # e.g. 20 credits for video generation

        # Step 2: Reserve Credits
        tx_id = await self.credits.reserve_credits(
            user_id=payload.user_id, job_id=job_id, required_credits=required_credits
        )

        try:
            # Step 3: Route to Video Adapter
            adapter_res = await self.routing.execute_with_routing_and_failover(
                capability="video",
                payload={"context_payload": resolved_payload},
            )

            video_url = adapter_res.get("video_url", "")
            provider_response = adapter_res.get("provider_response", {})
            final_prompt = adapter_res.get("final_prompt", "")

            # Step 4: Storage & CDN Upload (Cloudinary)
            cloudinary_result = self.storage.upload_video(video_url=video_url, job_id=job_id)
            cdn_video_url = cloudinary_result.get("secure_url") or video_url

            # Update Job Record
            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.VIDEO_COMPLETED,
                    "final_video_prompt": final_prompt,
                    "video_url": video_url,
                    "cdn_video_url": cdn_video_url,
                    "cloudinary_asset": cloudinary_result,
                    "provider_response": provider_response,
                },
            )

            # Step 5: Commit Credits
            await self.credits.commit_credits(
                user_id=payload.user_id,
                transaction_id=tx_id,
                generation_id=job_id,
                credits=required_credits,
            )

            # Step 6: Audit Monitoring Log
            await self.monitoring.log_generation(
                user_id=payload.user_id,
                generation_type="video",
                provider=adapter_res.get("provider", "fal_ai"),
                model="kling-video",
                credits=required_credits,
                status="completed",
                asset_id=cloudinary_result.get("public_id") or job_id,
            )

            return VideoGenerationResponse(
                job_id=job_id,
                video_url=video_url,
                cdn_video_url=cdn_video_url,
                cloudinary_asset=cloudinary_result,
                provider_response=provider_response,
                status=JobStatus.VIDEO_COMPLETED,
            )
        except Exception as exc:
            # Refund credits on failure
            await self.credits.refund_credits(
                user_id=payload.user_id,
                transaction_id=tx_id,
                generation_id=job_id,
                credits=required_credits,
                reason=str(exc),
            )
            await self.jobs.mark_failed(job_id, str(exc))
            raise
