from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from app.core.config import get_settings
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
from app.services.magica_service import MagicaVideoService
from app.services.openai_service import OpenAIScriptService

logger = logging.getLogger(__name__)

GenerationCallable = Callable[[str, bool], Awaitable[tuple[str, dict[str, Any], str]]]
ImageGenerationCallable = Callable[[str], Awaitable[tuple[str, dict[str, Any], str]]]


class MediaGenerationService:
    def __init__(self) -> None:
        self.contents = ContentRepository()
        self.generations = GenerationRepository()
        self.settings = get_settings()
        self.openai = OpenAIScriptService()
        self.fal = FalVideoService()
        self.magica = MagicaVideoService()
        self.storage = CloudinaryService()
        self.costs = CostService()

    async def create_image_prompt(
        self, payload: ImageGenerationPromptRequest
    ) -> ImageGenerationPromptResponse:
        settings = {
            "resolution": self._normalize_resolution(payload.resolution),
            "aspect_ratio": self._normalize_aspect_ratio(payload.aspect_ratio),
        }
        plan = await self.openai.plan_media_generation(
            user_prompt=payload.prompt,
            generation_type=AIGenerationType.IMAGE_GENERATION.value,
            settings=settings,
        )
        settings = self._apply_agentic_settings_plan(settings, plan)
        refine = await self.openai.refine_generation_prompt(
            user_prompt=self._prompt_with_agentic_guidance(payload.prompt, plan),
            generation_type=AIGenerationType.IMAGE_GENERATION.value,
            settings=settings,
        )
        openai_cost_total = round(refine["cost"] + plan["cost"], 8)
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.IMAGE_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "agentic_plan": plan,
                "openai_cost_total": openai_cost_total,
                "last_openai_cost": openai_cost_total,
                "last_openai_cost_formatted": self.costs.format_usd(openai_cost_total),
                "last_openai_usage": self._combine_openai_usage(
                    plan.get("usage"), refine.get("usage")
                ),
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._image_prompt_response(
            content,
            openai_cost=self.costs.format_usd(openai_cost_total),
        )

    async def create_video_prompt(
        self, payload: VideoGenerationPromptRequest
    ) -> VideoGenerationPromptResponse:
        settings = {
            "resolution": self._normalize_resolution(payload.resolution),
            "aspect_ratio": self._normalize_aspect_ratio(payload.aspect_ratio),
            "time": payload.time,
            "audio": payload.audio,
        }
        plan = await self.openai.plan_media_generation(
            user_prompt=payload.prompt,
            generation_type=AIGenerationType.VIDEO_GENERATION.value,
            settings=settings,
        )
        settings = self._apply_agentic_settings_plan(settings, plan)
        refine = await self.openai.refine_generation_prompt(
            user_prompt=self._prompt_with_agentic_guidance(payload.prompt, plan),
            generation_type=AIGenerationType.VIDEO_GENERATION.value,
            settings=settings,
        )
        openai_cost_total = round(refine["cost"] + plan["cost"], 8)
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.VIDEO_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "agentic_plan": plan,
                "openai_cost_total": openai_cost_total,
                "last_openai_cost": openai_cost_total,
                "last_openai_cost_formatted": self.costs.format_usd(openai_cost_total),
                "last_openai_usage": self._combine_openai_usage(
                    plan.get("usage"), refine.get("usage")
                ),
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._video_prompt_response(
            content,
            openai_cost=self.costs.format_usd(openai_cost_total),
        )

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
        plan = await self.openai.plan_media_generation(
            user_prompt=prompt,
            generation_type=AIGenerationType.VIDEO_EDIT.value,
            settings=settings,
            references=references,
        )
        settings = self._apply_agentic_settings_plan(settings, plan)
        refine = await self.openai.refine_generation_prompt(
            user_prompt=self._prompt_with_agentic_guidance(prompt, plan),
            generation_type=AIGenerationType.VIDEO_EDIT.value,
            settings=settings,
            references=references,
        )
        openai_cost_total = round(refine["cost"] + plan["cost"], 8)
        content_id = await self.contents.create_content(
            {
                "user_id": user_id,
                "type": AIGenerationType.VIDEO_EDIT.value,
                "user_prompt": prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": references,
                "agentic_plan": plan,
                "openai_cost_total": openai_cost_total,
                "last_openai_cost": openai_cost_total,
                "last_openai_cost_formatted": self.costs.format_usd(openai_cost_total),
                "last_openai_usage": self._combine_openai_usage(
                    plan.get("usage"), refine.get("usage")
                ),
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(user_id, content_id)
        return self._video_edit_prompt_response(
            content,
            openai_cost=self.costs.format_usd(openai_cost_total),
        )

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
        content = await self._require_typed_content(
            user_id, content_id, AIGenerationType.VIDEO_EDIT
        )
        return self._video_edit_prompt_detail_response(content)

    async def regenerate_image_prompt(
        self, user_id: str, content_id: str
    ) -> ImageGenerationPromptDetailResponse:
        content = await self._regenerate_prompt(
            user_id, content_id, AIGenerationType.IMAGE_GENERATION
        )
        return self._image_prompt_detail_response(content)

    async def regenerate_video_prompt(
        self, user_id: str, content_id: str
    ) -> VideoGenerationPromptDetailResponse:
        content = await self._regenerate_prompt(
            user_id, content_id, AIGenerationType.VIDEO_GENERATION
        )
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

        image_url, provider_response, model, provider = await self._agentic_generate_image(
            prompt=content["ai_refined_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            plan=content.get("agentic_plan") or settings.get("agentic_plan"),
        )
        storage_result = self.storage.upload_image(image_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or image_url
        fal_cost = self.costs.fal_image_cost(provider_response=provider_response)

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider=provider,
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

        video_url, provider_response, model, provider = await self._agentic_generate_video(
            prompt=content["ai_refined_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            time_seconds=time_seconds,
            audio=audio,
            plan=content.get("agentic_plan") or settings.get("agentic_plan"),
        )

        provider_cost = self.costs.fal_video_cost(
            time_seconds,
            provider_response=provider_response,
            audio=self._selected_agentic_audio(provider_response, default=audio),
        )
        if provider == "magica":
            provider_cost = float(provider_response.get("estimated_cost_usd") or provider_cost)

        storage_result = self.storage.upload_video(video_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or video_url
        self._cleanup_uploaded_local_source(video_url, storage_result)

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider=provider,
            model=model,
            fal_cost=provider_cost,
            storage_result=storage_result,
        )

    def _cleanup_uploaded_local_source(
        self, file_source: str, storage_result: dict[str, Any]
    ) -> None:
        if file_source.startswith(("http://", "https://")):
            return
        if not storage_result.get("is_uploaded"):
            return
        if not os.path.exists(file_source):
            return

        parent_dir = os.path.dirname(file_source)
        if os.path.basename(parent_dir).startswith(("jctranz_video_", "jctranz_magica_video_")):
            shutil.rmtree(parent_dir, ignore_errors=True)
        else:
            try:
                os.remove(file_source)
            except OSError:
                pass

    async def edit_video(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        content = await self._require_typed_content(
            payload.user_id, payload.content_id, AIGenerationType.VIDEO_EDIT
        )
        settings = content.get("settings") or {}
        references = content.get("references") or {}

        video_url, provider_response, model, provider = await self._agentic_edit_video(
            prompt=content["ai_refined_prompt"],
            video_ref=references.get("video_ref") or "",
            image_ref=references.get("image_ref"),
            audio=bool(settings.get("audio", True)),
            plan=content.get("agentic_plan") or settings.get("agentic_plan"),
        )
        storage_result = self.storage.upload_video(video_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or video_url
        fal_cost = self.costs.fal_video_edit_cost(provider_response=provider_response)

        return await self._save_generation_response(
            user_id=payload.user_id,
            content=content,
            content_url=content_url,
            provider_response=provider_response,
            provider=provider,
            model=model,
            fal_cost=fal_cost,
            storage_result=storage_result,
        )

    async def _agentic_generate_image(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        plan: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str, str]:
        attempts: list[dict[str, Any]] = []
        provider_calls: list[tuple[str, ImageGenerationCallable]] = []

        if self.settings.magica_image_provider_enabled and self.magica.is_configured:
            provider_calls.append(
                (
                    "magica",
                    lambda attempt_prompt: self.magica.generate_image_from_prompt(
                        prompt=attempt_prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                    ),
                )
            )

        if self.fal.api_key:
            provider_calls.append(
                (
                    "fal_ai",
                    lambda attempt_prompt: self.fal.generate_image_from_prompt(
                        prompt=attempt_prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                    ),
                )
            )

        if not provider_calls:
            raise RuntimeError("No image provider is configured.")

        safe_prompt = self._build_agentic_safe_prompt(
            prompt,
            generation_type=AIGenerationType.IMAGE_GENERATION.value,
        )
        provider_calls = self._sort_provider_calls(provider_calls, plan)
        variants = self._planned_prompt_variants(
            prompt=prompt,
            safe_prompt=safe_prompt,
            plan=plan,
        )

        last_error: Exception | None = None
        for provider, generate in provider_calls:
            for variant, attempt_prompt in variants:
                try:
                    image_url, provider_response, model = await generate(attempt_prompt)
                    attempts.append(
                        {
                            "provider": provider,
                            "variant": variant,
                            "status": "completed",
                        }
                    )
                    provider_response = self._with_agentic_metadata(
                        provider_response,
                        self._agentic_success_metadata(
                            attempts=attempts,
                            selected_provider=provider,
                            selected_variant=variant,
                            plan=plan,
                        ),
                    )
                    return image_url, provider_response, model, provider
                except Exception as exc:
                    last_error = exc
                    attempts.append(self._agentic_failed_attempt(provider, variant, exc))
                    logger.warning(
                        "Agentic image attempt failed provider=%s variant=%s: %s",
                        provider,
                        variant,
                        exc,
                    )
                    if variant == "original" and not self._should_retry_with_safe_prompt(exc):
                        break

        raise RuntimeError(
            "Agentic image generation failed after all provider attempts: "
            f"{self._format_exception(last_error)}"
        )

    async def _agentic_generate_video(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        time_seconds: int,
        audio: bool,
        plan: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str, str]:
        attempts: list[dict[str, Any]] = []
        provider_calls: list[tuple[str, GenerationCallable]] = []

        if self.settings.magica_video_provider_enabled and self.magica.is_configured:
            provider_calls.append(
                (
                    "magica",
                    lambda attempt_prompt, attempt_audio: self.magica.generate_video_from_prompt(
                        prompt=attempt_prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        time_seconds=time_seconds,
                        audio=attempt_audio,
                    ),
                )
            )

        if self.fal.api_key:
            provider_calls.append(
                (
                    "fal_ai",
                    lambda attempt_prompt, attempt_audio: self.fal.generate_video_from_prompt(
                        prompt=attempt_prompt,
                        resolution=resolution,
                        aspect_ratio=aspect_ratio,
                        time_seconds=time_seconds,
                        audio=attempt_audio,
                    ),
                )
            )

        if not provider_calls:
            raise RuntimeError("No video provider is configured.")

        safe_prompt = self._build_agentic_safe_prompt(
            prompt,
            generation_type=AIGenerationType.VIDEO_GENERATION.value,
        )
        provider_calls = self._sort_provider_calls(provider_calls, plan)
        variants = [
            (variant, variant_prompt, self._planned_variant_audio(variant, audio, plan))
            for variant, variant_prompt in self._planned_prompt_variants(
                prompt=prompt,
                safe_prompt=safe_prompt,
                plan=plan,
            )
        ]

        last_error: Exception | None = None
        for provider, generate in provider_calls:
            for variant, attempt_prompt, attempt_audio in variants:
                try:
                    video_url, provider_response, model = await generate(
                        attempt_prompt,
                        attempt_audio,
                    )
                    attempts.append(
                        {
                            "provider": provider,
                            "variant": variant,
                            "audio": attempt_audio,
                            "status": "completed",
                        }
                    )
                    provider_response = self._with_agentic_metadata(
                        provider_response,
                        self._agentic_success_metadata(
                            attempts=attempts,
                            selected_provider=provider,
                            selected_variant=variant,
                            plan=plan,
                        ),
                    )
                    return video_url, provider_response, model, provider
                except Exception as exc:
                    last_error = exc
                    attempts.append(
                        self._agentic_failed_attempt(provider, variant, exc, audio=attempt_audio)
                    )
                    logger.warning(
                        "Agentic video attempt failed provider=%s variant=%s: %s",
                        provider,
                        variant,
                        exc,
                    )
                    if variant == "original" and not self._should_retry_with_safe_prompt(exc):
                        break

        raise RuntimeError(
            "Agentic video generation failed after all provider attempts: "
            f"{self._format_exception(last_error)}"
        )

    async def _agentic_edit_video(
        self,
        *,
        prompt: str,
        video_ref: str,
        image_ref: str | None,
        audio: bool,
        plan: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str, str]:
        provider_calls: list[tuple[str, GenerationCallable]] = []

        if self.settings.magica_video_edit_provider_enabled and self.magica.is_configured:
            provider_calls.append(
                (
                    "magica",
                    lambda attempt_prompt, attempt_audio: self.magica.edit_video_from_prompt(
                        prompt=attempt_prompt,
                        video_ref=video_ref,
                        image_ref=image_ref,
                        audio=attempt_audio,
                    ),
                )
            )

        if self.fal.api_key:
            provider_calls.append(
                (
                    "fal_ai",
                    lambda attempt_prompt, attempt_audio: self.fal.edit_video_from_prompt(
                        prompt=attempt_prompt,
                        video_ref=video_ref,
                        image_ref=image_ref,
                        audio=attempt_audio,
                    ),
                )
            )

        if not provider_calls:
            raise RuntimeError("No video edit provider is configured.")

        safe_prompt = self._build_agentic_safe_prompt(
            prompt,
            generation_type=AIGenerationType.VIDEO_EDIT.value,
        )
        attempts: list[dict[str, Any]] = []
        provider_calls = self._sort_provider_calls(provider_calls, plan)
        variants = [
            (variant, variant_prompt, self._planned_variant_audio(variant, audio, plan))
            for variant, variant_prompt in self._planned_prompt_variants(
                prompt=prompt,
                safe_prompt=safe_prompt,
                plan=plan,
            )
        ]

        last_error: Exception | None = None
        for provider, edit in provider_calls:
            for variant, attempt_prompt, attempt_audio in variants:
                try:
                    video_url, provider_response, model = await edit(
                        attempt_prompt,
                        attempt_audio,
                    )
                    attempts.append(
                        {
                            "provider": provider,
                            "variant": variant,
                            "audio": attempt_audio,
                            "status": "completed",
                        }
                    )
                    provider_response = self._with_agentic_metadata(
                        provider_response,
                        self._agentic_success_metadata(
                            attempts=attempts,
                            selected_provider=provider,
                            selected_variant=variant,
                            plan=plan,
                        ),
                    )
                    return video_url, provider_response, model, provider
                except Exception as exc:
                    last_error = exc
                    attempts.append(
                        self._agentic_failed_attempt(provider, variant, exc, audio=attempt_audio)
                    )
                    logger.warning(
                        "Agentic video edit attempt failed provider=%s variant=%s: %s",
                        provider,
                        variant,
                        exc,
                    )
                    if variant == "original" and not self._should_retry_with_safe_prompt(exc):
                        break

        raise RuntimeError(
            "Agentic video edit failed after all attempts: "
            f"{self._format_exception(last_error)}"
        )

    def _apply_agentic_settings_plan(
        self,
        settings: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(settings)
        overrides = plan.get("settings_overrides")
        if not isinstance(overrides, dict):
            return merged

        if isinstance(overrides.get("resolution"), str):
            merged["resolution"] = self._normalize_resolution(overrides["resolution"])
        if isinstance(overrides.get("aspect_ratio"), str):
            merged["aspect_ratio"] = self._normalize_aspect_ratio(overrides["aspect_ratio"])
        if isinstance(overrides.get("audio"), bool):
            merged["audio"] = overrides["audio"]
        return merged

    def _settings_for_agentic_plan(self, settings: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(settings)
        if isinstance(normalized.get("resolution"), str):
            normalized["resolution"] = self._normalize_resolution(normalized["resolution"])
        if isinstance(normalized.get("aspect_ratio"), str):
            normalized["aspect_ratio"] = self._normalize_aspect_ratio(
                normalized["aspect_ratio"]
            )
        return normalized

    def _combine_openai_usage(
        self,
        first: dict[str, Any] | None,
        second: dict[str, Any] | None,
    ) -> dict[str, Any]:
        first = first or {}
        second = second or {}
        input_tokens = int(first.get("input_tokens") or 0) + int(
            second.get("input_tokens") or 0
        )
        output_tokens = int(first.get("output_tokens") or 0) + int(
            second.get("output_tokens") or 0
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _prompt_with_agentic_guidance(self, prompt: str, plan: dict[str, Any]) -> str:
        guidance = plan.get("prompt_guidance")
        if not isinstance(guidance, str) or not guidance.strip():
            return prompt
        return f"{prompt}\n\nPlanner guidance for the media model:\n{guidance.strip()}"

    def _sort_provider_calls(
        self,
        provider_calls: list[tuple[str, Any]],
        plan: dict[str, Any] | None,
    ) -> list[tuple[str, Any]]:
        preferred_provider = self._planned_preferred_provider(plan)
        return sorted(
            provider_calls,
            key=lambda item: 0 if item[0] == preferred_provider else 1,
        )

    def _planned_preferred_provider(self, plan: dict[str, Any] | None) -> str:
        if not isinstance(plan, dict):
            return "magica"
        preferred_provider = plan.get("preferred_provider")
        return preferred_provider if preferred_provider in {"magica", "fal_ai"} else "magica"

    def _planned_prompt_variants(
        self,
        *,
        prompt: str,
        safe_prompt: str,
        plan: dict[str, Any] | None,
    ) -> list[tuple[str, str]]:
        retry_strategy = plan.get("retry_strategy") if isinstance(plan, dict) else None
        if not isinstance(retry_strategy, list):
            retry_strategy = ["original", "safe_prompt"]

        variants: list[tuple[str, str]] = []
        for item in retry_strategy:
            if item == "original":
                variants.append(("original", prompt))
            elif item == "safe_prompt":
                variants.append(("safe_prompt", safe_prompt))

        if not variants:
            variants.append(("original", prompt))
        if not any(variant == "safe_prompt" for variant, _ in variants):
            variants.append(("safe_prompt", safe_prompt))
        return variants[:2]

    def _planned_variant_audio(
        self,
        variant: str,
        original_audio: bool,
        plan: dict[str, Any] | None,
    ) -> bool:
        if variant == "safe_prompt" and original_audio:
            return False
        if not isinstance(plan, dict):
            return original_audio

        overrides = plan.get("settings_overrides")
        if isinstance(overrides, dict) and isinstance(overrides.get("audio"), bool):
            return overrides["audio"]
        return original_audio

    def _build_agentic_safe_prompt(self, prompt: str, *, generation_type: str) -> str:
        if generation_type == AIGenerationType.VIDEO_EDIT.value:
            return (
                f"{prompt}\n\n"
                "Apply only the requested edit to the provided reference video. Keep the result "
                "policy-safe, non-graphic, non-sexual, and free of real-person impersonation. "
                "Keep people generic and non-identifiable when present. Do not add logos, "
                "captions, watermarks, or unrelated new content. Preserve the original timing, "
                "composition, and camera motion as much as possible."
            )

        media_label = (
            "image" if generation_type == AIGenerationType.IMAGE_GENERATION.value else "video"
        )
        return (
            f"{prompt}\n\n"
            f"Generate a policy-safe {media_label}. Use generic, non-identifiable people if any "
            "people appear. Avoid graphic violence, sexual content, weapons, drugs, hate symbols, "
            "celebrity likenesses, copyrighted characters, brand logos, readable text, captions, "
            "and watermarks. Preserve the original creative intent, scene, mood, lighting, and "
            "camera direction."
        )

    def _normalize_resolution(self, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "")
        aliases = {
            "420p": "480p",
            "480": "480p",
            "720": "720p",
            "1080": "1080p",
        }
        return aliases.get(normalized, normalized or value.strip())

    def _normalize_aspect_ratio(self, value: str) -> str:
        normalized = value.strip().replace(" ", "")
        aliases = {
            "landscape": "16:9",
            "portrait": "9:16",
            "square": "1:1",
        }
        return aliases.get(normalized.lower(), normalized or value.strip())

    def _should_retry_with_safe_prompt(self, exc: Exception) -> bool:
        message = self._format_exception(exc).lower()
        retry_signals = (
            "content_policy",
            "policy",
            "moderation",
            "safety",
            "unsafe",
            "bad gateway",
            "provider",
            "failed",
            "timeout",
            "timed out",
            "did not return",
        )
        return any(signal in message for signal in retry_signals)

    def _agentic_failed_attempt(
        self,
        provider: str,
        variant: str,
        exc: Exception,
        *,
        audio: bool | None = None,
    ) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "provider": provider,
            "variant": variant,
            "status": "failed",
            "error": self._format_exception(exc),
        }
        if audio is not None:
            attempt["audio"] = audio
        return attempt

    def _agentic_success_metadata(
        self,
        *,
        attempts: list[dict[str, Any]],
        selected_provider: str,
        selected_variant: str,
        plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "enabled": True,
            "planner": "llm" if isinstance(plan, dict) else "rule_based",
            "selected_provider": selected_provider,
            "selected_variant": selected_variant,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        if isinstance(plan, dict):
            metadata["plan"] = {
                "task_type": plan.get("task_type"),
                "preferred_provider": plan.get("preferred_provider"),
                "risk_level": plan.get("risk_level"),
                "retry_strategy": plan.get("retry_strategy"),
                "prompt_guidance": plan.get("prompt_guidance"),
            }
        return metadata

    def _with_agentic_metadata(
        self,
        provider_response: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        response = provider_response if isinstance(provider_response, dict) else {}
        return {
            **response,
            "agentic": metadata,
        }

    def _selected_agentic_audio(self, provider_response: dict[str, Any], *, default: bool) -> bool:
        agentic = provider_response.get("agentic")
        if not isinstance(agentic, dict):
            return default

        selected_provider = agentic.get("selected_provider")
        selected_variant = agentic.get("selected_variant")
        attempts = agentic.get("attempts")
        if not isinstance(attempts, list):
            return default

        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            if (
                attempt.get("provider") == selected_provider
                and attempt.get("variant") == selected_variant
                and isinstance(attempt.get("audio"), bool)
            ):
                return attempt["audio"]
        return default

    def _format_exception(self, exc: Exception | None) -> str:
        if exc is None:
            return "unknown error"
        message = str(exc).strip()
        return message[-1200:] if message else exc.__class__.__name__

    async def _regenerate_prompt(
        self, user_id: str, content_id: str, generation_type: AIGenerationType
    ) -> dict[str, Any]:
        content = await self._require_typed_content(user_id, content_id, generation_type)
        settings = self._settings_for_agentic_plan(content.get("settings") or {})
        references = content.get("references") or {}
        plan = await self.openai.plan_media_generation(
            user_prompt=content["user_prompt"],
            generation_type=generation_type.value,
            settings=settings,
            references=references,
        )
        settings = self._apply_agentic_settings_plan(settings, plan)
        refine = await self.openai.refine_generation_prompt(
            user_prompt=self._prompt_with_agentic_guidance(content["user_prompt"], plan),
            generation_type=generation_type.value,
            settings=settings,
            references=references,
            previous_refined_prompt=content.get("ai_refined_prompt"),
        )
        openai_cost = round(refine["cost"] + plan["cost"], 8)
        updated = await self.contents.replace_refined_prompt(
            user_id=user_id,
            content_id=content_id,
            new_refined_prompt=refine["refined_prompt"],
            openai_cost=openai_cost,
            openai_cost_formatted=self.costs.format_usd(openai_cost),
            openai_usage=self._combine_openai_usage(plan.get("usage"), refine.get("usage")),
            agentic_plan=plan,
            settings=settings,
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
