import json

from fastapi import HTTPException

from app.core.config import get_settings
from app.models.ai_models import (
    JobStatus,
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    ScriptRegenerationRequest,
    ScriptScene,
    VideoGenerationContext,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from app.prompts.script_prompt import build_script_regeneration_prompt
from app.services.cloudinary_service import CloudinaryService
from app.services.fal_service import FalVideoService
from app.services.job_repository import JobRepository
from app.services.openai_service import OpenAIScriptService


class AIWorkflowService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.jobs = JobRepository()
        self.openai = OpenAIScriptService()
        self.fal = FalVideoService()
        self.cloudinary = CloudinaryService()

    def _primary_job_image(self, job: dict) -> str | None:
        product_image_data_urls = job.get("product_image_data_urls")
        if isinstance(product_image_data_urls, list) and product_image_data_urls:
            first_data_url = product_image_data_urls[0]
            if isinstance(first_data_url, str) and first_data_url:
                return first_data_url

        product_image_url = job.get("product_image_url")
        if isinstance(product_image_url, str) and product_image_url:
            return product_image_url

        product_images = job.get("product_images")
        if isinstance(product_images, list) and product_images:
            first_image = product_images[0]
            if isinstance(first_image, str) and first_image:
                return first_image

        return None

    def _script_scenes_to_text(self, scenes: list[dict] | list[ScriptScene] | str | None) -> str:
        if not scenes:
            return ""
        if isinstance(scenes, str):
            return scenes

        lines: list[str] = []
        for scene in scenes:
            if isinstance(scene, ScriptScene):
                scene_data = scene.model_dump()
            elif isinstance(scene, dict):
                scene_data = scene
            else:
                continue
            sequence = scene_data.get("sequence", "")
            time = scene_data.get("time", "")
            visual = scene_data.get("visual", "")
            voiceover = scene_data.get("voiceover", "")
            lines.append(f"{sequence}. [{time}] {visual} | VO: {voiceover}")
        return "\n".join(lines)

    async def generate_script(
        self,
        payload: ScriptGenerationRequest,
        product_image_data_urls: list[str] | None = None,
    ) -> ScriptGenerationResponse:
        job_id = await self.jobs.create_job(
            {
                "status": JobStatus.PENDING,
                "user_id": payload.user_id,
                "product_name": payload.product_name,
                "product_description": payload.product_description,
                "product_images": payload.product_images,
                "product_image_data_urls": product_image_data_urls,
                "product_image_url": product_image_data_urls[0]
                if product_image_data_urls
                else None,
                "time_seconds": payload.time_seconds,
            }
        )

        try:
            script = await self.openai.generate_promotional_script(payload)
            scenes = script.get("scenes")
            if not isinstance(scenes, list) or not scenes:
                raise RuntimeError("OpenAI did not return a valid scenes array.")
            await self.jobs.update_job(
                job_id,
                {
                    "status": JobStatus.SCRIPT_GENERATED,
                    "promotional_script": scenes,
                },
            )
            return ScriptGenerationResponse(job_id=job_id, promotional_script=scenes)
        except Exception as exc:
            await self.jobs.mark_failed(job_id, str(exc))
            raise

    async def regenerate_script(self, payload: ScriptRegenerationRequest) -> ScriptGenerationResponse:
        job = await self.jobs.get_job(payload.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="AI workflow job not found.")

        if job.get("user_id") != payload.user_id:
            raise HTTPException(status_code=403, detail="This job does not belong to the user.")

        product_name = job.get("product_name")
        product_description = job.get("product_description")
        product_images = job.get("product_images")
        time_seconds = job.get("time_seconds") or self.settings.fal_video_duration_seconds

        if not product_name:
            raise HTTPException(status_code=400, detail="Saved job is missing product_name.")
        if not product_description:
            raise HTTPException(
                status_code=400,
                detail="Saved job is missing product_description.",
            )

        prompt = build_script_regeneration_prompt(
            product_name=product_name,
            product_description=product_description,
            product_images=product_images if isinstance(product_images, list) else None,
            time_seconds=time_seconds,
            current_script=json.dumps(
                [scene.model_dump() for scene in payload.promotional_script], ensure_ascii=False
            ),
        )

        try:
            regenerated_script = await self.openai.generate_text_from_prompt(prompt)
            regenerated_scenes = regenerated_script.get("scenes")
            if not isinstance(regenerated_scenes, list) or not regenerated_scenes:
                raise RuntimeError("OpenAI did not return a valid scenes array.")
            script_history = job.get("script_history")
            if isinstance(script_history, list):
                updated_history = [*script_history, [scene.model_dump() for scene in payload.promotional_script]]
            else:
                updated_history = [[scene.model_dump() for scene in payload.promotional_script]]

            await self.jobs.update_job(
                payload.job_id,
                {
                    "status": JobStatus.SCRIPT_GENERATED,
                    "promotional_script": regenerated_scenes,
                    "script_history": updated_history,
                },
            )
            return ScriptGenerationResponse(
                job_id=payload.job_id,
                promotional_script=regenerated_scenes,
            )
        except Exception as exc:
            await self.jobs.mark_failed(payload.job_id, str(exc))
            raise

    async def generate_video(self, payload: VideoGenerationRequest) -> VideoGenerationResponse:
        job = await self.jobs.get_job(payload.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="AI workflow job not found.")

        if job.get("user_id") != payload.user_id:
            raise HTTPException(status_code=403, detail="This job does not belong to the user.")

        product_name = job.get("product_name")
        approved_script = self._script_scenes_to_text(job.get("promotional_script"))
        product_image_url = self._primary_job_image(job)
        time_seconds = job.get("time_seconds") or self.settings.fal_video_duration_seconds

        if not product_name:
            raise HTTPException(status_code=400, detail="Saved job is missing product_name.")
        if not approved_script:
            raise HTTPException(status_code=400, detail="Saved job is missing the promotional script.")

        resolved_payload = VideoGenerationContext(
            product_name=product_name,
            product_image_url=product_image_url,
            approved_script=approved_script,
            time_seconds=time_seconds,
        )

        job_id = payload.job_id
        await self.jobs.update_job(
            job_id,
            {
                "status": JobStatus.VIDEO_GENERATING,
                "product_name": product_name,
                "product_image_url": product_image_url,
                "approved_script_text": approved_script,
                "time_seconds": time_seconds,
                "video_generation_mode": "image-to-video" if product_image_url else "text-to-video",
            },
        )

        try:
            video_url, provider_response, final_prompt = await self.fal.generate_video(
                resolved_payload
            )

            # Build Cloudinary Asset (Storage Step 4)
            cloudinary_result = self.cloudinary.upload_video(video_url=video_url, job_id=job_id)
            cdn_video_url = cloudinary_result.get("secure_url") or video_url

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
            return VideoGenerationResponse(
                job_id=job_id,
                video_url=video_url,
                cdn_video_url=cdn_video_url,
                cloudinary_asset=cloudinary_result,
                provider_response=provider_response,
            )
        except Exception as exc:
            await self.jobs.mark_failed(job_id, str(exc))
            raise

    async def get_job(self, job_id: str):
        return await self.jobs.get_job(job_id)
