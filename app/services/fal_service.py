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
        self.image_model = settings.fal_video_model
        self.text_model = settings.fal_text_video_model
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

        final_prompt = build_video_prompt(payload)
        duration_seconds = payload.time_seconds or self.default_duration_seconds
        arguments = {
            "prompt": final_prompt,
            "duration": str(duration_seconds),
        }
        model = self.text_model

        if payload.product_image_url:
            model = self.image_model
            arguments["image_url"] = payload.product_image_url

        result = await asyncio.to_thread(
            fal_client.subscribe,
            model,
            arguments=arguments,
            with_logs=True,
        )

        video_url = self._extract_video_url(result)
        if not video_url:
            raise RuntimeError("fal.AI did not return a video_url.")

        provider_response = result if isinstance(result, dict) else {"result": result}
        provider_response["model"] = model
        provider_response["generation_mode"] = (
            "image-to-video" if payload.product_image_url else "text-to-video"
        )

        return video_url, provider_response, final_prompt

    def _extract_video_url(self, result: Any) -> str | None:
        if isinstance(result, str):
            if result.startswith(("http://", "https://")):
                return result
            return None

        if isinstance(result, list):
            for item in result:
                video_url = self._extract_video_url(item)
                if video_url:
                    return video_url
            return None

        if not isinstance(result, dict):
            return None

        if isinstance(result.get("video_url"), str):
            return result["video_url"]

        if isinstance(result.get("url"), str):
            return result["url"]

        video = result.get("video")
        video_url = self._extract_video_url(video)
        if video_url:
            return video_url

        videos = result.get("videos")
        videos_url = self._extract_video_url(videos)
        if videos_url:
            return videos_url

        output = result.get("output")
        output_url = self._extract_video_url(output)
        if output_url:
            return output_url

        return None
