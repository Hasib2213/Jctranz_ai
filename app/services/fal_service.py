import asyncio
import os
from typing import Any

import fal_client

from app.core.config import get_settings
from app.models.ai_models import VideoGenerationContext
from app.prompts.video_prompt import build_video_prompt


class FalVideoService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.fal_video_model
        self.api_key = settings.fal_key
        self.default_duration_seconds = settings.fal_video_duration_seconds
        os.environ["FAL_KEY"] = self.api_key

    async def generate_video(self, payload: VideoGenerationContext) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("FAL_KEY is not configured.")
        if not payload.product_name:
            raise RuntimeError("product_name is required to generate a video.")
        if not payload.approved_script:
            raise RuntimeError("approved_script is required to generate a video.")
        if not payload.product_image_url:
            raise RuntimeError("product_image_url is required to generate a video.")

        final_prompt = build_video_prompt(payload)
        duration_seconds = payload.time_seconds or self.default_duration_seconds
        arguments = {
            "prompt": final_prompt,
            "image_url": str(payload.product_image_url),
            "duration": str(duration_seconds),
        }

        result = await asyncio.to_thread(
            fal_client.subscribe,
            self.model,
            arguments=arguments,
            with_logs=True,
        )

        video_url = self._extract_video_url(result)
        if not video_url:
            raise RuntimeError("fal.AI did not return a video_url.")

        return video_url, dict(result), final_prompt

    def _extract_video_url(self, result: dict[str, Any]) -> str | None:
        if isinstance(result.get("video_url"), str):
            return result["video_url"]

        video = result.get("video")
        if isinstance(video, dict) and isinstance(video.get("url"), str):
            return video["url"]

        videos = result.get("videos")
        if isinstance(videos, list) and videos:
            first_video = videos[0]
            if isinstance(first_video, dict) and isinstance(first_video.get("url"), str):
                return first_video["url"]

        output = result.get("output")
        if isinstance(output, dict):
            return self._extract_video_url(output)

        return None
