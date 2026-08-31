from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.models.ai_models import (
    AIGenerationType,
    ImageGenerationPromptDetailResponse,
    ImageGenerationPromptRequest,
    ImageGenerationPromptResponse,
    MediaGenerationRequest,
    MediaGenerationResponse,
    PromptVersion,
    VideoEditPromptDetailResponse,
    VideoEditPromptResponse,
    VideoGenerationPromptDetailResponse,
    VideoGenerationPromptRequest,
    VideoGenerationPromptResponse,
)
from app.services.cloudinary_service import CloudinaryService
from app.services.content_repository import ContentRepository
from app.services.cost_service import CostService
from app.services.fal_service import FalVideoService
from app.services.generation_repository import GenerationRepository
from app.services.openai_service import OpenAIScriptService


class MediaGenerationService:
    def __init__(self) -> None:
        self.contents = ContentRepository()
        self.generations = GenerationRepository()
        self.openai = OpenAIScriptService()
        self.fal = FalVideoService()
        self.storage = CloudinaryService()
        self.costs = CostService()

    async def create_image_prompt(
        self, payload: ImageGenerationPromptRequest
    ) -> ImageGenerationPromptResponse:
        settings = {
            "resolution": payload.resolution,
            "aspect_ratio": payload.aspect_ratio,
        }
        refine = await self.openai.refine_generation_prompt(
            user_prompt=payload.prompt,
            generation_type=AIGenerationType.IMAGE_GENERATION.value,
            settings=settings,
        )
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.IMAGE_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "openai_cost_total": refine["cost"],
                "last_openai_cost": refine["cost"],
                "last_openai_cost_formatted": refine["formatted_cost"],
                "last_openai_usage": refine["usage"],
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._image_prompt_response(content, openai_cost=refine["formatted_cost"])

    async def create_video_prompt(
        self, payload: VideoGenerationPromptRequest
    ) -> VideoGenerationPromptResponse:
        settings = {
            "resolution": payload.resolution,
            "aspect_ratio": payload.aspect_ratio,
            "time": payload.time,
            "audio": payload.audio,
        }
        refine = await self.openai.refine_generation_prompt(
            user_prompt=payload.prompt,
            generation_type=AIGenerationType.VIDEO_GENERATION.value,
            settings=settings,
        )
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.VIDEO_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "openai_cost_total": refine["cost"],
                "last_openai_cost": refine["cost"],
                "last_openai_cost_formatted": refine["formatted_cost"],
                "last_openai_usage": refine["usage"],
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._video_prompt_response(content, openai_cost=refine["formatted_cost"])

    async def create_video_edit_prompt(
        self,
        *,
        user_id: str,
        prompt: str,
        video_ref: str,
        image_ref: str | None,
        audio: bool,
    ) -> VideoEditPromptResponse:
        if not video_ref:
            raise HTTPException(status_code=400, detail="video_ref is required.")

        settings = {"audio": audio}
        references = {
            "video_ref": video_ref,
            "image_ref": image_ref,
        }
        refine = await self.openai.refine_generation_prompt(
            user_prompt=prompt,
            generation_type=AIGenerationType.VIDEO_EDIT.value,
            settings=settings,
            references=references,
        )
        content_id = await self.contents.create_content(
            {
                "user_id": user_id,
                "type": AIGenerationType.VIDEO_EDIT.value,
                "user_prompt": prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": references,
                "openai_cost_total": refine["cost"],
                "last_openai_cost": refine["cost"],
                "last_openai_cost_formatted": refine["formatted_cost"],
                "last_openai_usage": refine["usage"],
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(user_id, content_id)
        return self._video_edit_prompt_response(content, openai_cost=refine["formatted_cost"])

    async def get_image_prompt(
        self, user_id: str, content_id: str
    ) -> ImageGenerationPromptDetailResponse:
        content = await self._require_typed_content(
            user_id, content_id, AIGenerationType.IMAGE_GENERATION
        )
        return self._image_prompt_detail_response(content)

    async def get_video_prompt(
        self, user_id: str, content_id: str
    ) -> VideoGenerationPromptDetailResponse:
        content = await self._require_typed_content(
            user_id, content_id, AIGenerationType.VIDEO_GENERATION
        )
        return self._video_prompt_detail_response(content)

    async def get_video_edit_prompt(
        self, user_id: str, content_id: str
    ) -> VideoEditPromptDetailResponse:
        content = await self._require_typed_content(user_id, content_id, AIGenerationType.VIDEO_EDIT)
        return self._video_edit_prompt_detail_response(content)

    async def regenerate_image_prompt(
        self, user_id: str, content_id: str
    ) -> ImageGenerationPromptDetailResponse:
        content = await self._regenerate_prompt(user_id, content_id, AIGenerationType.IMAGE_GENERATION)
        return self._image_prompt_detail_response(content)

    async def regenerate_video_prompt(
        self, user_id: str, content_id: str
    ) -> VideoGenerationPromptDetailResponse:
        content = await self._regenerate_prompt(user_id, content_id, AIGenerationType.VIDEO_GENERATION)
        return self._video_prompt_detail_response(content)

    async def regenerate_video_edit_prompt(
        self, user_id: str, content_id: str
    ) -> VideoEditPromptDetailResponse:
        content = await self._regenerate_prompt(user_id, content_id, AIGenerationType.VIDEO_EDIT)
        return self._video_edit_prompt_detail_response(content)

    async def generate_image(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        content = await self._require_typed_content(
            payload.user_id, payload.content_id, AIGenerationType.IMAGE_GENERATION
        )
        settings = content.get("settings") or {}

        image_url, provider_response, model = await self.fal.generate_image_from_prompt(
            prompt=content["ai_refined_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
        )
        storage_result = self.storage.upload_image(image_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or image_url
        fal_cost = self.costs.fal_image_cost(provider_response=provider_response)

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider="fal_ai",
            model=model,
            fal_cost=fal_cost,
            storage_result=storage_result,
        )

    async def generate_video(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        content = await self._require_typed_content(
            payload.user_id, payload.content_id, AIGenerationType.VIDEO_GENERATION
        )
        settings = content.get("settings") or {}
        time_seconds = int(settings.get("time") or 5)
        audio = bool(settings.get("audio", True))

        video_url, provider_response, model = await self.fal.generate_video_from_prompt(
            prompt=content["ai_refined_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            time_seconds=time_seconds,
            audio=audio,
        )
        storage_result = self.storage.upload_video(video_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or video_url
        fal_cost = self.costs.fal_video_cost(
            time_seconds,
            provider_response=provider_response,
            audio=audio,
        )

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider="fal_ai",
            model=model,
            fal_cost=fal_cost,
            storage_result=storage_result,
        )

    async def edit_video(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        content = await self._require_typed_content(
            payload.user_id, payload.content_id, AIGenerationType.VIDEO_EDIT
        )
        settings = content.get("settings") or {}
        references = content.get("references") or {}

        video_url, provider_response, model = await self.fal.edit_video_from_prompt(
            prompt=content["ai_refined_prompt"],
            video_ref=references.get("video_ref") or "",
            image_ref=references.get("image_ref"),
            audio=bool(settings.get("audio", True)),
        )
        storage_result = self.storage.upload_video(video_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or video_url
        fal_cost = self.costs.fal_video_edit_cost(provider_response=provider_response)

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider="fal_ai",
            model=model,
            fal_cost=fal_cost,
            storage_result=storage_result,
        )

    async def _regenerate_prompt(
        self, user_id: str, content_id: str, generation_type: AIGenerationType
    ) -> dict[str, Any]:
        content = await self._require_typed_content(user_id, content_id, generation_type)
        refine = await self.openai.refine_generation_prompt(
            user_prompt=content["user_prompt"],
            generation_type=generation_type.value,
            settings=content.get("settings") or {},
            references=content.get("references") or {},
            previous_refined_prompt=content.get("ai_refined_prompt"),
        )
        updated = await self.contents.replace_refined_prompt(
            user_id=user_id,
            content_id=content_id,
            new_refined_prompt=refine["refined_prompt"],
            openai_cost=refine["cost"],
            openai_cost_formatted=refine["formatted_cost"],
            openai_usage=refine["usage"],
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Content prompt not found.")
        return updated

    async def _save_generation_response(
        self,
        *,
        user_id: str,
        content: dict[str, Any],
        content_url: str,
        provider_response: dict[str, Any],
        provider: str,
        model: str,
        fal_cost: float,
        storage_result: dict[str, Any],
    ) -> MediaGenerationResponse:
        openai_cost = float(content.get("openai_cost_total") or 0.0)
        total_cost = round(openai_cost + fal_cost, 8)
        generated_content_id = await self.generations.create_generation(
            {
                "user_id": user_id,
                "content_id": content["content_id"],
                "type": content["type"],
                "status": "completed",
                "content_url": content_url,
                "provider": provider,
                "model": model,
                "openai_cost": openai_cost,
                "falai_cost": fal_cost,
                "total_cost": total_cost,
                "provider_response": provider_response,
                "storage_result": storage_result,
            }
        )
        return MediaGenerationResponse(
            user_id=user_id,
            generated_content_id=generated_content_id,
            content_url=content_url,
            openAI_cost=self.costs.format_usd(openai_cost),
            falAI_cost=self.costs.format_usd(fal_cost),
            total_cost=self.costs.format_usd(total_cost),
        )

    async def _require_content(self, user_id: str, content_id: str) -> dict[str, Any]:
        content = await self.contents.get_content(user_id=user_id, content_id=content_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Content prompt not found.")
        return content

    async def _require_typed_content(
        self, user_id: str, content_id: str, generation_type: AIGenerationType
    ) -> dict[str, Any]:
        content = await self._require_content(user_id=user_id, content_id=content_id)
        if content.get("type") != generation_type.value:
            raise HTTPException(status_code=400, detail="Content type does not match endpoint.")
        return content

    def _versions(self, content: dict[str, Any]) -> list[PromptVersion]:
        versions = content.get("prompt_versions")
        if not isinstance(versions, list):
            return []
        return [PromptVersion.model_validate(version) for version in versions]

    def _image_prompt_response(
        self, content: dict[str, Any], openai_cost: str | None = None
    ) -> ImageGenerationPromptResponse:
        settings = content.get("settings") or {}
        return ImageGenerationPromptResponse(
            user_id=content["user_id"],
            content_id=content["content_id"],
            AI_refine_prompt=content["ai_refined_prompt"],
            user_prompt=content["user_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            openAI_cost=openai_cost or self.costs.format_usd(content.get("last_openai_cost")),
        )

    def _image_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> ImageGenerationPromptDetailResponse:
        base = self._image_prompt_response(content)
        return ImageGenerationPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openAI_cost_total=self.costs.format_usd(content.get("openai_cost_total")),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )

    def _video_prompt_response(
        self, content: dict[str, Any], openai_cost: str | None = None
    ) -> VideoGenerationPromptResponse:
        settings = content.get("settings") or {}
        return VideoGenerationPromptResponse(
            user_id=content["user_id"],
            content_id=content["content_id"],
            AI_refine_prompt=content["ai_refined_prompt"],
            user_prompt=content["user_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            time=int(settings.get("time") or 0),
            audio=bool(settings.get("audio", True)),
            openAI_cost=openai_cost or self.costs.format_usd(content.get("last_openai_cost")),
        )

    def _video_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> VideoGenerationPromptDetailResponse:
        base = self._video_prompt_response(content)
        return VideoGenerationPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openAI_cost_total=self.costs.format_usd(content.get("openai_cost_total")),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )

    def _video_edit_prompt_response(
        self, content: dict[str, Any], openai_cost: str | None = None
    ) -> VideoEditPromptResponse:
        settings = content.get("settings") or {}
        references = content.get("references") or {}
        return VideoEditPromptResponse(
            user_id=content["user_id"],
            content_id=content["content_id"],
            AI_refine_prompt=content["ai_refined_prompt"],
            user_prompt=content["user_prompt"],
            video_ref=references.get("video_ref") or "",
            image_ref=references.get("image_ref"),
            audio=bool(settings.get("audio", True)),
            openAI_cost=openai_cost or self.costs.format_usd(content.get("last_openai_cost")),
        )

    def _video_edit_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> VideoEditPromptDetailResponse:
        base = self._video_edit_prompt_response(content)
        return VideoEditPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openAI_cost_total=self.costs.format_usd(content.get("openai_cost_total")),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )
