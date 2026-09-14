from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
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
    OpenAITokenUsage,
    PromptVersion,
    VideoEditPromptDetailResponse,
    VideoEditPromptResponse,
    VideoGenerationPromptDetailResponse,
    VideoGenerationPromptRequest,
    VideoGenerationPromptResponse,
)
from app.core.redis import get_arq_redis
from app.services.cloudinary_service import CloudinaryService


from app.services.content_repository import ContentRepository
from app.services.cost_service import CostService
from app.services.fal_service import FalVideoService
from app.services.generation_repository import GenerationRepository
from app.services.magica_service import MagicaVideoService
from app.services.openai_service import OpenAIScriptService
from app.services.video_segment_service import VideoSegmentService

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
        self.video_segments = VideoSegmentService()

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
        combined_usage = self._combine_openai_usage(plan.get("usage"), refine.get("usage"))
        clean_usage = {
            "input_tokens": int(combined_usage.get("input_tokens") or 0),
            "output_tokens": int(combined_usage.get("output_tokens") or 0),
            "total_tokens": int(combined_usage.get("total_tokens") or 0),
        }
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.IMAGE_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "agentic_plan": plan,
                "openai_tokens": clean_usage,
                "openai_tokens_total": clean_usage,
                "last_openai_usage": clean_usage,
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._image_prompt_response(content, openai_tokens=clean_usage)

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
        combined_usage = self._combine_openai_usage(plan.get("usage"), refine.get("usage"))
        clean_usage = {
            "input_tokens": int(combined_usage.get("input_tokens") or 0),
            "output_tokens": int(combined_usage.get("output_tokens") or 0),
            "total_tokens": int(combined_usage.get("total_tokens") or 0),
        }
        content_id = await self.contents.create_content(
            {
                "user_id": payload.user_id,
                "type": AIGenerationType.VIDEO_GENERATION.value,
                "user_prompt": payload.prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": {},
                "agentic_plan": plan,
                "openai_tokens": clean_usage,
                "openai_tokens_total": clean_usage,
                "last_openai_usage": clean_usage,
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(payload.user_id, content_id)
        return self._video_prompt_response(content, openai_tokens=clean_usage)

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
        combined_usage = self._combine_openai_usage(plan.get("usage"), refine.get("usage"))
        clean_usage = {
            "input_tokens": int(combined_usage.get("input_tokens") or 0),
            "output_tokens": int(combined_usage.get("output_tokens") or 0),
            "total_tokens": int(combined_usage.get("total_tokens") or 0),
        }
        content_id = await self.contents.create_content(
            {
                "user_id": user_id,
                "type": AIGenerationType.VIDEO_EDIT.value,
                "user_prompt": prompt,
                "ai_refined_prompt": refine["refined_prompt"],
                "settings": settings,
                "references": references,
                "agentic_plan": plan,
                "openai_tokens": clean_usage,
                "openai_tokens_total": clean_usage,
                "last_openai_usage": clean_usage,
                "openai_model": refine["model"],
            }
        )
        content = await self._require_content(user_id, content_id)
        return self._video_edit_prompt_response(content, openai_tokens=clean_usage)

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
        pool = await get_arq_redis()
        if pool is not None and self.settings.arq_enabled:
            try:
                job = await pool.enqueue_job("task_generate_image", payload.user_id, payload.content_id)
                if job is not None:
                    result = await job.result(timeout=self.settings.arq_job_timeout, poll_delay=1.0)
                    return MediaGenerationResponse.model_validate(result)
            except Exception as exc:
                logger.warning("ARQ job execution failed (%s), falling back to direct execution.", exc)

        return await self._execute_direct_image_generation(payload)

    async def _execute_direct_image_generation(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
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
        pool = await get_arq_redis()
        if pool is not None and self.settings.arq_enabled:
            try:
                job = await pool.enqueue_job("task_generate_video", payload.user_id, payload.content_id)
                if job is not None:
                    result = await job.result(timeout=self.settings.arq_job_timeout, poll_delay=1.0)
                    return MediaGenerationResponse.model_validate(result)
            except Exception as exc:
                logger.warning("ARQ job execution failed (%s), falling back to direct execution.", exc)

        return await self._execute_direct_video_generation(payload)

    async def _execute_direct_video_generation(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
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
        temp_prefixes = (
            "jctranz_video_",
            "jctranz_magica_video_",
            "jctranz_segment_edit_",
        )
        if os.path.basename(parent_dir).startswith(temp_prefixes):
            shutil.rmtree(parent_dir, ignore_errors=True)
        else:
            try:
                os.remove(file_source)
            except OSError:
                pass

    async def edit_video(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        pool = await get_arq_redis()
        if pool is not None and self.settings.arq_enabled:
            try:
                job = await pool.enqueue_job("task_generate_video_edit", payload.user_id, payload.content_id)
                if job is not None:
                    result = await job.result(timeout=self.settings.arq_job_timeout, poll_delay=1.0)
                    return MediaGenerationResponse.model_validate(result)
            except Exception as exc:
                logger.warning("ARQ job execution failed (%s), falling back to direct execution.", exc)

        return await self._execute_direct_video_edit(payload)

    async def _execute_direct_video_edit(self, payload: MediaGenerationRequest) -> MediaGenerationResponse:
        content = await self._require_typed_content(
            payload.user_id, payload.content_id, AIGenerationType.VIDEO_EDIT
        )
        settings = content.get("settings") or {}
        references = content.get("references") or {}
        video_ref = references.get("video_ref") or ""
        image_ref = references.get("image_ref")
        audio = bool(settings.get("audio", True))
        plan = content.get("agentic_plan") or settings.get("agentic_plan")

        segment_result = await self._try_agentic_segment_edit_video(
            content_id=payload.content_id,
            user_prompt=content["user_prompt"],
            prompt=content["ai_refined_prompt"],
            video_ref=video_ref,
            image_ref=image_ref,
            audio=audio,
            plan=plan,
        )
        if segment_result is None:
            provider_temp_dir = tempfile.mkdtemp(prefix="jctranz_provider_edit_")
            try:
                provider_video_ref, preflight_metadata = await self._prepare_video_edit_input(
                    video_ref=video_ref,
                    content_id=payload.content_id,
                    temp_dir=provider_temp_dir,
                    suffix="direct",
                    audio=audio,
                )
                video_url, provider_response, model, provider = await self._agentic_edit_video(
                    prompt=content["ai_refined_prompt"],
                    video_ref=provider_video_ref,
                    image_ref=image_ref,
                    audio=audio,
                    plan=plan,
                )
                provider_response = self._with_video_input_preflight_metadata(
                    provider_response,
                    preflight_metadata,
                )
            finally:
                shutil.rmtree(provider_temp_dir, ignore_errors=True)
        else:
            video_url, provider_response, model, provider = segment_result

        storage_result = self.storage.upload_video(video_url, job_id=payload.content_id)
        content_url = storage_result.get("secure_url") or video_url
        self._cleanup_uploaded_local_source(video_url, storage_result)
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

    async def _try_agentic_segment_edit_video(
        self,
        *,
        content_id: str,
        user_prompt: str,
        prompt: str,
        video_ref: str,
        image_ref: str | None,
        audio: bool,
        plan: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], str, str] | None:
        if not video_ref:
            return None

        temp_dir = tempfile.mkdtemp(prefix="jctranz_segment_edit_")
        try:
            source_path = await asyncio.to_thread(
                self.video_segments.prepare_source,
                video_ref,
                temp_dir,
                "source_video",
            )
            duration = await asyncio.to_thread(self.video_segments.probe_duration, source_path)
            if duration <= self.video_segments.max_direct_edit_seconds:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None

            analysis = None
            use_global_candidate = self._should_use_global_timeline_edit(
                user_prompt=user_prompt,
                prompt=prompt,
                duration=duration,
                image_ref=image_ref,
            )
            if use_global_candidate:
                analysis = await self._analyze_edit_segment(
                    user_prompt=user_prompt,
                    prompt=prompt,
                    source_path=source_path,
                    temp_dir=temp_dir,
                    duration=duration,
                )
                target_ranges = self._target_ranges_from_analysis(analysis, duration)
                if self._should_edit_full_timeline(
                    analysis=analysis,
                    target_ranges=target_ranges,
                    duration=duration,
                    user_prompt=user_prompt,
                    prompt=prompt,
                ):
                    return await self._edit_global_video_chunks(
                        content_id=content_id,
                        prompt=prompt,
                        source_path=source_path,
                        temp_dir=temp_dir,
                        duration=duration,
                        image_ref=image_ref,
                        audio=audio,
                        plan=plan,
                    )

                return await self._edit_selected_video_ranges(
                    content_id=content_id,
                    prompt=prompt,
                    source_path=source_path,
                    temp_dir=temp_dir,
                    duration=duration,
                    image_ref=image_ref,
                    audio=audio,
                    plan=plan,
                    analysis=analysis,
                    target_ranges=target_ranges,
                )

            if analysis is None:
                analysis = await self._analyze_edit_segment(
                    user_prompt=user_prompt,
                    prompt=prompt,
                    source_path=source_path,
                    temp_dir=temp_dir,
                    duration=duration,
                )
            if analysis.get("coverage") in {"localized", "partial", "throughout"}:
                target_ranges = self._target_ranges_from_analysis(analysis, duration)
                if self._should_edit_full_timeline(
                    analysis=analysis,
                    target_ranges=target_ranges,
                    duration=duration,
                    user_prompt=user_prompt,
                    prompt=prompt,
                ):
                    return await self._edit_global_video_chunks(
                        content_id=content_id,
                        prompt=prompt,
                        source_path=source_path,
                        temp_dir=temp_dir,
                        duration=duration,
                        image_ref=image_ref,
                        audio=audio,
                        plan=plan,
                    )

                return await self._edit_selected_video_ranges(
                    content_id=content_id,
                    prompt=prompt,
                    source_path=source_path,
                    temp_dir=temp_dir,
                    duration=duration,
                    image_ref=image_ref,
                    audio=audio,
                    plan=plan,
                    analysis=analysis,
                    target_ranges=target_ranges,
                )
            split_paths = await asyncio.to_thread(
                self.video_segments.split_for_segment_edit,
                source_path,
                temp_dir,
                start_time=float(analysis["start_time"]),
                end_time=float(analysis["end_time"]),
                preserve_audio=audio,
            )

            target_path = split_paths.get("target")
            if not target_path:
                raise RuntimeError("Video segment split did not produce a target clip.")

            target_ref, preflight_metadata = await self._prepare_video_edit_input(
                video_ref=target_path,
                content_id=content_id,
                temp_dir=temp_dir,
                suffix="target_segment",
                audio=audio,
            )
            edit_prompt = self._segment_edit_prompt(prompt, analysis)
            edited_url, provider_response, model, provider = await self._agentic_edit_video(
                prompt=edit_prompt,
                video_ref=target_ref,
                image_ref=image_ref,
                audio=audio,
                plan=plan,
            )
            edited_target_path = await asyncio.to_thread(
                self.video_segments.prepare_source,
                edited_url,
                temp_dir,
                "edited_target.mp4",
            )

            width, height = await asyncio.to_thread(
                self.video_segments.probe_dimensions,
                source_path,
            )
            clip_pairs = [
                (split_paths.get("before"), split_paths.get("before")),
                (edited_target_path, split_paths.get("target")),
                (split_paths.get("after"), split_paths.get("after")),
            ]
            clip_paths = [path for path, _ in clip_pairs if isinstance(path, str)]
            audio_source_paths = [
                audio_path if isinstance(audio_path, str) else None
                for path, audio_path in clip_pairs
                if isinstance(path, str)
            ]
            final_path = await asyncio.to_thread(
                self.video_segments.merge_clips,
                clip_paths,
                temp_dir,
                width=width,
                height=height,
                preserve_audio=audio,
                audio_source_paths=audio_source_paths,
            )

            provider_response = self._with_segment_edit_metadata(
                provider_response,
                {
                    "enabled": True,
                    "mode": "auto_detect_split_edit_merge",
                    "source_duration_seconds": round(duration, 3),
                    "max_direct_edit_seconds": self.video_segments.max_direct_edit_seconds,
                    "selected_time_range": {
                        "start_time": analysis["start_time"],
                        "end_time": analysis["end_time"],
                    },
                    "confidence": analysis.get("confidence"),
                    "target_object": analysis.get("target_object"),
                    "replacement_object": analysis.get("replacement_object"),
                    "reason": analysis.get("reason"),
                    "used_image_reference": bool(image_ref),
                    "preserved_audio": audio,
                    "audio_source": "original_segments_when_available" if audio else "none",
                    "provider_input_preflight": preflight_metadata,
                    "steps": [
                        "downloaded_source_video",
                        "analyzed_sampled_frames",
                        "split_before_target_after",
                        "edited_target_clip",
                        "merged_final_video",
                    ],
                },
            )
            return final_path, provider_response, f"segment_edit:{model}", provider
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.warning("Agentic segment edit failed; falling back to direct edit: %s", exc)
            return None

    async def _edit_global_video_chunks(
        self,
        *,
        content_id: str,
        prompt: str,
        source_path: str,
        temp_dir: str,
        duration: float,
        image_ref: str | None,
        audio: bool,
        plan: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any], str, str]:
        chunks = await asyncio.to_thread(
            self.video_segments.split_into_chunks,
            source_path,
            temp_dir,
            max_seconds=self.video_segments.max_direct_edit_seconds,
            preserve_audio=audio,
        )
        if not chunks:
            raise RuntimeError("Global segment edit did not produce any chunks.")

        width, height = await asyncio.to_thread(
            self.video_segments.probe_dimensions,
            source_path,
        )
        edited_paths: list[str] = []
        audio_source_paths: list[str | None] = []
        chunk_results: list[dict[str, Any]] = []
        selected_providers: list[str] = []
        selected_models: list[str] = []
        total_provider_cost = 0.0

        for chunk in chunks:
            chunk_index = int(chunk["index"])
            chunk_path = str(chunk["path"])
            provider_ref, preflight_metadata = await self._prepare_video_edit_input(
                video_ref=chunk_path,
                content_id=content_id,
                temp_dir=temp_dir,
                suffix=f"global_chunk_{chunk_index}",
                audio=audio,
            )
            chunk_prompt = self._global_segment_edit_prompt(
                prompt,
                chunk=chunk,
                chunk_count=len(chunks),
                has_image_reference=bool(image_ref),
            )
            edited_url, provider_response, model, provider = await self._agentic_edit_video(
                prompt=chunk_prompt,
                video_ref=provider_ref,
                image_ref=image_ref,
                audio=audio,
                plan=plan,
            )
            provider_cost = self.costs.extract_provider_cost(provider_response) or 0.0
            total_provider_cost += provider_cost
            edited_path = await asyncio.to_thread(
                self.video_segments.prepare_source,
                edited_url,
                temp_dir,
                f"edited_global_chunk_{chunk_index}.mp4",
            )
            edited_paths.append(edited_path)
            audio_source_paths.append(chunk_path)
            selected_providers.append(provider)
            selected_models.append(model)
            chunk_results.append(
                {
                    "index": chunk_index,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "duration_seconds": chunk["duration_seconds"],
                    "provider": provider,
                    "model": model,
                    "provider_input_preflight": preflight_metadata,
                    "provider_response": self._summarize_provider_response(provider_response),
                }
            )

        final_path = await asyncio.to_thread(
            self.video_segments.merge_clips,
            edited_paths,
            temp_dir,
            width=width,
            height=height,
            preserve_audio=audio,
            audio_source_paths=audio_source_paths,
        )
        unique_providers = sorted(set(selected_providers))
        provider = unique_providers[0] if len(unique_providers) == 1 else "agentic_multi_provider"
        model = "global_segment_edit:" + "+".join(dict.fromkeys(selected_models))
        provider_response = {
            "provider": provider,
            "model": model,
            "provider_cost": round(total_provider_cost, 8) if total_provider_cost else None,
            "agentic": {
                "enabled": True,
                "planner": "llm" if isinstance(plan, dict) else "rule_based",
                "global_segment_edit": {
                    "enabled": True,
                    "mode": "full_timeline_chunk_edit_merge",
                    "source_duration_seconds": round(duration, 3),
                    "chunk_count": len(chunks),
                    "max_chunk_seconds": self.video_segments.max_direct_edit_seconds,
                    "used_image_reference": bool(image_ref),
                    "preserved_audio": audio,
                    "audio_source": "original_chunks_when_available" if audio else "none",
                    "chunks": chunk_results,
                    "steps": [
                        "downloaded_source_video",
                        "detected_global_edit_intent",
                        "split_full_video_into_chunks",
                        "edited_each_chunk",
                        "merged_edited_chunks",
                    ],
                },
            },
        }
        return final_path, provider_response, model, provider

    async def _edit_selected_video_ranges(
        self,
        *,
        content_id: str,
        prompt: str,
        source_path: str,
        temp_dir: str,
        duration: float,
        image_ref: str | None,
        audio: bool,
        plan: dict[str, Any] | None,
        analysis: dict[str, Any],
        target_ranges: list[dict[str, float]],
    ) -> tuple[str, dict[str, Any], str, str]:
        parts = await asyncio.to_thread(
            self.video_segments.split_for_range_edits,
            source_path,
            temp_dir,
            ranges=target_ranges,
            preserve_audio=audio,
            max_edit_seconds=self.video_segments.max_direct_edit_seconds,
        )
        if not parts:
            raise RuntimeError("Selective segment edit did not produce any timeline parts.")

        width, height = await asyncio.to_thread(
            self.video_segments.probe_dimensions,
            source_path,
        )
        output_paths: list[str] = []
        audio_source_paths: list[str | None] = []
        part_results: list[dict[str, Any]] = []
        selected_providers: list[str] = []
        selected_models: list[str] = []
        total_provider_cost = 0.0

        for part in parts:
            part_index = int(part["index"])
            part_path = str(part["path"])
            audio_source_paths.append(part_path)

            if part["mode"] != "edit":
                output_paths.append(part_path)
                part_results.append(
                    {
                        "index": part_index,
                        "mode": "keep",
                        "start_time": part["start_time"],
                        "end_time": part["end_time"],
                        "duration_seconds": part["duration_seconds"],
                        "provider": None,
                    }
                )
                continue

            provider_ref, preflight_metadata = await self._prepare_video_edit_input(
                video_ref=part_path,
                content_id=content_id,
                temp_dir=temp_dir,
                suffix=f"selected_range_{part_index}",
                audio=audio,
            )
            part_prompt = self._selected_range_edit_prompt(
                prompt,
                part=part,
                has_image_reference=bool(image_ref),
            )
            edited_url, provider_response, model, provider = await self._agentic_edit_video(
                prompt=part_prompt,
                video_ref=provider_ref,
                image_ref=image_ref,
                audio=audio,
                plan=plan,
            )
            provider_cost = self.costs.extract_provider_cost(provider_response) or 0.0
            total_provider_cost += provider_cost
            edited_path = await asyncio.to_thread(
                self.video_segments.prepare_source,
                edited_url,
                temp_dir,
                f"edited_selected_range_{part_index}.mp4",
            )
            output_paths.append(edited_path)
            selected_providers.append(provider)
            selected_models.append(model)
            part_results.append(
                {
                    "index": part_index,
                    "mode": "edit",
                    "start_time": part["start_time"],
                    "end_time": part["end_time"],
                    "duration_seconds": part["duration_seconds"],
                    "provider": provider,
                    "model": model,
                    "provider_input_preflight": preflight_metadata,
                    "provider_response": self._summarize_provider_response(provider_response),
                }
            )

        final_path = await asyncio.to_thread(
            self.video_segments.merge_clips,
            output_paths,
            temp_dir,
            width=width,
            height=height,
            preserve_audio=audio,
            audio_source_paths=audio_source_paths,
        )
        unique_providers = sorted(set(selected_providers))
        provider = unique_providers[0] if len(unique_providers) == 1 else "agentic_multi_provider"
        model = "selective_segment_edit:" + "+".join(dict.fromkeys(selected_models))
        provider_response = {
            "provider": provider,
            "model": model,
            "provider_cost": round(total_provider_cost, 8) if total_provider_cost else None,
            "agentic": {
                "enabled": True,
                "planner": "llm" if isinstance(plan, dict) else "rule_based",
                "selective_segment_edit": {
                    "enabled": True,
                    "mode": "analyze_selective_ranges_edit_merge",
                    "source_duration_seconds": round(duration, 3),
                    "max_edit_seconds": self.video_segments.max_direct_edit_seconds,
                    "target_ranges": target_ranges,
                    "coverage": analysis.get("coverage"),
                    "confidence": analysis.get("confidence"),
                    "target_object": analysis.get("target_object"),
                    "replacement_object": analysis.get("replacement_object"),
                    "reason": analysis.get("reason"),
                    "used_image_reference": bool(image_ref),
                    "preserved_audio": audio,
                    "audio_source": "original_parts_when_available" if audio else "none",
                    "edited_part_count": len(selected_models),
                    "total_part_count": len(parts),
                    "parts": part_results,
                    "steps": [
                        "downloaded_source_video",
                        "analyzed_sampled_frames",
                        "selected_target_ranges",
                        "split_keep_and_edit_parts",
                        "edited_selected_parts",
                        "merged_timeline_parts",
                    ],
                },
            },
        }
        return final_path, provider_response, model, provider

    async def _prepare_video_edit_input(
        self,
        *,
        video_ref: str,
        content_id: str,
        temp_dir: str,
        suffix: str,
        audio: bool,
    ) -> tuple[str, dict[str, Any]]:
        if not video_ref:
            raise RuntimeError("video_ref is required for video editing.")

        source_path = await asyncio.to_thread(
            self.video_segments.prepare_source,
            video_ref,
            temp_dir,
            f"{suffix}_source",
        )
        normalized_path = os.path.join(temp_dir, f"{suffix}_provider_720p.mp4")
        metadata = await asyncio.to_thread(
            self.video_segments.normalize_for_provider,
            source_path,
            normalized_path,
            preserve_audio=audio,
        )
        upload = self.storage.upload_video(
            normalized_path,
            job_id=f"{content_id}_{suffix}_provider_input",
        )
        provider_video_ref = upload.get("secure_url") or normalized_path
        return provider_video_ref, {
            **metadata,
            "enabled": True,
            "source": "local" if os.path.exists(video_ref) else "remote",
            "provider_video_ref": provider_video_ref,
            "cloudinary_uploaded": bool(upload.get("is_uploaded")),
            "cloudinary_format": upload.get("format"),
        }

    async def _analyze_edit_segment(
        self,
        *,
        user_prompt: str,
        prompt: str,
        source_path: str,
        temp_dir: str,
        duration: float,
    ) -> dict[str, Any]:
        explicit_range = self._time_range_from_prompt(user_prompt, duration)
        if explicit_range is not None:
            start_time, end_time = explicit_range
            return {
                "requires_segment_edit": True,
                "target_object": "",
                "replacement_object": "",
                "start_time": start_time,
                "end_time": end_time,
                "coverage": "localized",
                "target_ranges": [
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ],
                "confidence": "high",
                "reason": "User prompt included an explicit time range.",
                "edit_prompt": prompt,
            }

        try:
            frames = await asyncio.to_thread(
                self.video_segments.extract_analysis_frames,
                source_path,
                temp_dir,
            )
            return await self.openai.analyze_video_edit_segment(
                user_prompt=user_prompt,
                refined_prompt=prompt,
                duration_seconds=duration,
                frames=frames,
                max_segment_seconds=self.video_segments.max_direct_edit_seconds,
            )
        except Exception as exc:
            logger.warning("LLM video segment analysis failed; using fallback range: %s", exc)
            return self._fallback_segment_analysis(user_prompt, prompt, duration)

    def _fallback_segment_analysis(
        self,
        user_prompt: str,
        prompt: str,
        duration: float,
    ) -> dict[str, Any]:
        start_time, end_time = self._time_range_from_prompt(user_prompt, duration) or (
            0.0,
            min(duration, self.video_segments.max_direct_edit_seconds),
        )
        return {
            "requires_segment_edit": True,
            "target_object": self._guess_replacement_side(user_prompt, side="target"),
            "replacement_object": self._guess_replacement_side(
                user_prompt,
                side="replacement",
            ),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "coverage": "unclear",
            "target_ranges": [
                {
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                }
            ],
            "confidence": "low",
            "reason": "Fallback selected the earliest provider-safe segment.",
            "edit_prompt": prompt,
        }

    def _time_range_from_prompt(self, prompt: str, duration: float) -> tuple[float, float] | None:
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?",
            r"from\s+(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?\s+to\s+"
            r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)?",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if not match:
                continue
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            if end_time <= start_time:
                continue
            start_time = max(0.0, min(start_time, duration))
            max_end = min(duration, start_time + self.video_segments.max_direct_edit_seconds)
            end_time = max(start_time, min(end_time, max_end))
            return round(start_time, 3), round(end_time, 3)
        return None

    def _guess_replacement_side(self, prompt: str, *, side: str) -> str:
        replace_match = re.search(
            r"replace\s+(?:the\s+)?([\w\s-]+?)\s+with\s+(?:a|an|the)?\s*([\w\s-]+)",
            prompt,
            flags=re.IGNORECASE,
        )
        if not replace_match:
            return ""
        group_index = 1 if side == "target" else 2
        return replace_match.group(group_index).strip(" .,!?:;")

    def _segment_edit_prompt(self, prompt: str, analysis: dict[str, Any]) -> str:
        edit_prompt = analysis.get("edit_prompt")
        if not isinstance(edit_prompt, str) or not edit_prompt.strip():
            edit_prompt = prompt
        return (
            f"{edit_prompt.strip()}\n\n"
            "This is only the selected target segment from a longer source video. Apply the "
            "requested edit only inside this segment. Preserve the original camera movement, "
            "background, lighting, framing, timing, and all non-target objects so this clip can "
            "be merged back into the original video without a visible jump."
        )

    def _global_segment_edit_prompt(
        self,
        prompt: str,
        *,
        chunk: dict[str, Any],
        chunk_count: int,
        has_image_reference: bool,
    ) -> str:
        reference_text = (
            " Use the provided reference image as the exact visual replacement target."
            if has_image_reference
            else ""
        )
        return (
            f"{prompt.strip()}\n\n"
            f"This is chunk {int(chunk['index']) + 1} of {chunk_count} from a longer video "
            f"covering {chunk['start_time']}s to {chunk['end_time']}s. Apply the requested "
            "edit throughout this entire chunk wherever the target appears, including every "
            f"visible frame and angle.{reference_text} Preserve the original hands, pose, "
            "camera motion, background, lighting, timing, and all non-target details so all "
            "edited chunks can be merged into one continuous video."
        )

    def _selected_range_edit_prompt(
        self,
        prompt: str,
        *,
        part: dict[str, Any],
        has_image_reference: bool,
    ) -> str:
        reference_text = (
            " Use the provided reference image as the exact visual replacement target."
            if has_image_reference
            else ""
        )
        return (
            f"{prompt.strip()}\n\n"
            f"This clip is only the selected part from {part['start_time']}s to "
            f"{part['end_time']}s of a longer source video. Apply the requested edit throughout "
            f"this entire clip wherever the target appears.{reference_text} Preserve the original "
            "hands, pose, camera motion, background, lighting, timing, and all non-target details "
            "so this edited part can be merged back with unchanged video parts."
        )

    def _target_ranges_from_analysis(
        self,
        analysis: dict[str, Any],
        duration: float,
    ) -> list[dict[str, float]]:
        raw_ranges = analysis.get("target_ranges")
        candidates: list[dict[str, float]] = []
        if isinstance(raw_ranges, list):
            for item in raw_ranges:
                if not isinstance(item, dict):
                    continue
                try:
                    start_time = float(item.get("start_time", 0.0))
                    end_time = float(item.get("end_time", 0.0))
                except (TypeError, ValueError):
                    continue
                start_time = max(0.0, min(start_time, duration))
                end_time = max(start_time, min(end_time, duration))
                if end_time - start_time >= 0.25:
                    candidates.append(
                        {
                            "start_time": round(start_time, 3),
                            "end_time": round(end_time, 3),
                        }
                    )

        if not candidates:
            candidates = [
                {
                    "start_time": float(analysis.get("start_time") or 0.0),
                    "end_time": float(
                        analysis.get("end_time")
                        or min(duration, self.video_segments.max_direct_edit_seconds)
                    ),
                }
            ]
        return self._merge_edit_ranges(candidates, duration)

    def _merge_edit_ranges(
        self,
        ranges: list[dict[str, float]],
        duration: float,
    ) -> list[dict[str, float]]:
        normalized: list[dict[str, float]] = []
        for item in ranges:
            try:
                start_time = float(item["start_time"])
                end_time = float(item["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            start_time = max(0.0, min(start_time, duration))
            end_time = max(start_time, min(end_time, duration))
            if end_time - start_time >= 0.25:
                normalized.append(
                    {
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3),
                    }
                )

        normalized.sort(key=lambda item: item["start_time"])
        merged: list[dict[str, float]] = []
        for item in normalized:
            if not merged or item["start_time"] > merged[-1]["end_time"] + 0.25:
                merged.append(dict(item))
                continue
            merged[-1]["end_time"] = max(merged[-1]["end_time"], item["end_time"])
        return merged

    def _should_edit_full_timeline(
        self,
        *,
        analysis: dict[str, Any],
        target_ranges: list[dict[str, float]],
        duration: float,
        user_prompt: str,
        prompt: str,
    ) -> bool:
        coverage = analysis.get("coverage")
        if coverage == "throughout":
            return True
        if coverage in {"localized", "partial"}:
            return self._range_coverage_fraction(target_ranges, duration) >= 0.7
        if self._has_explicit_global_marker(f"{user_prompt}\n{prompt}"):
            return True
        return self._range_coverage_fraction(target_ranges, duration) >= 0.8

    def _range_coverage_fraction(
        self,
        ranges: list[dict[str, float]],
        duration: float,
    ) -> float:
        if duration <= 0:
            return 0.0
        merged = self._merge_edit_ranges(ranges, duration)
        covered = sum(item["end_time"] - item["start_time"] for item in merged)
        return max(0.0, min(1.0, covered / duration))

    def _should_use_global_timeline_edit(
        self,
        *,
        user_prompt: str,
        prompt: str,
        duration: float,
        image_ref: str | None,
    ) -> bool:
        combined_prompt = f"{user_prompt}\n{prompt}"
        if (
            self._time_range_from_prompt(combined_prompt, duration) is not None
            or self._has_local_time_edit_cue(combined_prompt)
        ):
            return False

        text = combined_prompt.lower()
        global_markers = (
            "throughout",
            "entire video",
            "whole video",
            "full video",
            "all video",
            "everywhere",
            "every frame",
            "all frames",
            "sob jaigai",
            "shob jaigai",
            "sobar jaigai",
            "পুরো",
            "সব জায়গা",
            "সব জায়গা",
        )
        if any(marker in text for marker in global_markers):
            return True

        edit_verbs = (
            "replace",
            "swap",
            "change",
            "turn",
            "make",
            "remove",
            "convert",
            "substitute",
        )
        has_globalish_context = (
            " in the video" in text
            or "current " in text
            or "currently" in text
            or "same, just" in text
            or bool(image_ref)
        )
        return has_globalish_context and any(verb in text for verb in edit_verbs)

    def _has_explicit_global_marker(self, prompt: str) -> bool:
        text = prompt.lower()
        global_markers = (
            "throughout",
            "entire video",
            "whole video",
            "full video",
            "all video",
            "everywhere",
            "every frame",
            "all frames",
            "whole timeline",
            "sob jaigai",
            "shob jaigai",
            "sobar jaigai",
            "puro video",
        )
        return any(marker in text for marker in global_markers)

    def _has_local_time_edit_cue(self, prompt: str) -> bool:
        time_cue_patterns = (
            r"\bfirst\s+\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b",
            r"\blast\s+\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b",
            r"\bbeginning\s+\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b",
            r"\bending\s+\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)\b",
            r"\bintro\b",
            r"\boutro\b",
        )
        return any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in time_cue_patterns)

    def _summarize_provider_response(self, provider_response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(provider_response, dict):
            return {"raw_type": type(provider_response).__name__}

        summary_keys = (
            "provider",
            "node_type",
            "run_id",
            "status",
            "creditUsed",
            "video_url",
            "keep_audio",
            "request_model",
            "cost",
            "provider_cost",
            "total_cost",
        )
        summary = {
            key: provider_response[key]
            for key in summary_keys
            if key in provider_response
        }
        agentic = provider_response.get("agentic")
        if isinstance(agentic, dict):
            summary["agentic"] = {
                key: agentic.get(key)
                for key in (
                    "selected_provider",
                    "selected_variant",
                    "attempt_count",
                    "planner",
                )
                if key in agentic
            }
        return summary

    def _with_segment_edit_metadata(
        self,
        provider_response: dict[str, Any],
        segment_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        response = provider_response if isinstance(provider_response, dict) else {}
        agentic = response.get("agentic")
        if not isinstance(agentic, dict):
            agentic = {"enabled": True, "planner": "llm"}
        agentic = {
            **agentic,
            "segment_edit": segment_metadata,
        }
        return {
            **response,
            "agentic": agentic,
        }

    def _with_video_input_preflight_metadata(
        self,
        provider_response: dict[str, Any],
        preflight_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        response = provider_response if isinstance(provider_response, dict) else {}
        agentic = response.get("agentic")
        if not isinstance(agentic, dict):
            agentic = {"enabled": True}
        agentic = {
            **agentic,
            "video_input_preflight": preflight_metadata,
        }
        return {
            **response,
            "agentic": agentic,
        }

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
        combined_usage = self._combine_openai_usage(plan.get("usage"), refine.get("usage"))
        clean_usage = {
            "input_tokens": int(combined_usage.get("input_tokens") or 0),
            "output_tokens": int(combined_usage.get("output_tokens") or 0),
            "total_tokens": int(combined_usage.get("total_tokens") or 0),
        }
        updated = await self.contents.replace_refined_prompt(
            user_id=user_id,
            content_id=content_id,
            new_refined_prompt=refine["refined_prompt"],
            openai_usage=clean_usage,
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
        openai_tokens = self._total_token_usage_from_content(content)
        magica_credits = self.costs.extract_magica_credits(
            provider_response, fallback_cost_usd=fal_cost
        )
        generated_content_id = await self.generations.create_generation(
            {
                "user_id": user_id,
                "content_id": content["content_id"],
                "type": content["type"],
                "status": "completed",
                "content_url": content_url,
                "provider": provider,
                "model": model,
                "openai_tokens": openai_tokens.model_dump(),
                "magica_credits": magica_credits,
                "falai_cost": fal_cost,
                "provider_response": provider_response,
                "storage_result": storage_result,
            }
        )

        return MediaGenerationResponse(
            user_id=user_id,
            generated_content_id=generated_content_id,
            content_url=content_url,
            provider=provider,
            openai_tokens=openai_tokens,
            magica_credits=magica_credits,
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

    def _token_usage_from_content(
        self, content: dict[str, Any], override: dict[str, Any] | None = None
    ) -> OpenAITokenUsage:
        raw = override or content.get("last_openai_usage") or content.get("openai_tokens") or {}
        if not isinstance(raw, dict):
            raw = {}
        return OpenAITokenUsage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    def _total_token_usage_from_content(self, content: dict[str, Any]) -> OpenAITokenUsage:
        raw = (
            content.get("openai_tokens_total")
            or content.get("last_openai_usage")
            or content.get("openai_tokens")
            or {}
        )
        if not isinstance(raw, dict):
            raw = {}
        return OpenAITokenUsage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            total_tokens=int(raw.get("total_tokens") or 0),
        )

    def _versions(self, content: dict[str, Any]) -> list[PromptVersion]:
        versions = content.get("prompt_versions")
        if not isinstance(versions, list):
            return []
        cleaned: list[PromptVersion] = []
        for version in versions:
            if not isinstance(version, dict):
                continue
            raw_tokens = (
                version.get("openai_tokens")
                or version.get("openai_usage")
                or {}
            )
            if not isinstance(raw_tokens, dict):
                raw_tokens = {}
            tokens = OpenAITokenUsage(
                input_tokens=int(raw_tokens.get("input_tokens") or 0),
                output_tokens=int(raw_tokens.get("output_tokens") or 0),
                total_tokens=int(raw_tokens.get("total_tokens") or 0),
            )
            cleaned.append(
                PromptVersion(
                    version=int(version.get("version") or 1),
                    AI_refine_prompt=str(
                        version.get("AI_refine_prompt")
                        or version.get("ai_refined_prompt")
                        or ""
                    ),
                    openai_tokens=tokens,
                    created_at=version.get("created_at"),
                )
            )
        return cleaned

    def _image_prompt_response(
        self, content: dict[str, Any], openai_tokens: dict[str, Any] | None = None
    ) -> ImageGenerationPromptResponse:
        settings = content.get("settings") or {}
        return ImageGenerationPromptResponse(
            user_id=content["user_id"],
            content_id=content["content_id"],
            AI_refine_prompt=content["ai_refined_prompt"],
            user_prompt=content["user_prompt"],
            resolution=settings.get("resolution", ""),
            aspect_ratio=settings.get("aspect_ratio", ""),
            openai_tokens=self._token_usage_from_content(content, openai_tokens),
        )

    def _image_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> ImageGenerationPromptDetailResponse:
        base = self._image_prompt_response(content)
        return ImageGenerationPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openai_tokens_total=self._total_token_usage_from_content(content),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )

    def _video_prompt_response(
        self, content: dict[str, Any], openai_tokens: dict[str, Any] | None = None
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
            openai_tokens=self._token_usage_from_content(content, openai_tokens),
        )

    def _video_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> VideoGenerationPromptDetailResponse:
        base = self._video_prompt_response(content)
        return VideoGenerationPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openai_tokens_total=self._total_token_usage_from_content(content),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )

    def _video_edit_prompt_response(
        self, content: dict[str, Any], openai_tokens: dict[str, Any] | None = None
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
            openai_tokens=self._token_usage_from_content(content, openai_tokens),
        )

    def _video_edit_prompt_detail_response(
        self, content: dict[str, Any]
    ) -> VideoEditPromptDetailResponse:
        base = self._video_edit_prompt_response(content)
        return VideoEditPromptDetailResponse(
            **base.model_dump(by_alias=True),
            prompt_versions=self._versions(content),
            openai_tokens_total=self._total_token_usage_from_content(content),
            created_at=content.get("created_at"),
            updated_at=content.get("updated_at"),
        )

