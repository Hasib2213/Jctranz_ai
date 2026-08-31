import asyncio
import os
import tempfile
import subprocess
import urllib.request
import logging
import re
from typing import Any

import fal_client

from app.core.config import get_settings
from app.models.ai_models import VideoGenerationContext
from app.prompts.video_prompt import build_video_prompt

logger = logging.getLogger(__name__)


def _calculate_clip_durations(total_seconds: int) -> list[str]:
    if total_seconds <= 5:
        return ["5"]
    elif total_seconds <= 10:
        return ["10"]
    elif total_seconds == 15:
        return ["10", "5"]
    elif total_seconds == 20:
        return ["10", "10"]
    elif total_seconds == 30:
        return ["10", "10", "10"]
    else:
        full_clips = total_seconds // 10
        rem = total_seconds % 10
        clips = ["10"] * full_clips
        if rem >= 5:
            clips.append("5")
        return clips or ["10"]


def _normalize_image_size(resolution: str) -> str | dict[str, int]:
    resolution = resolution.strip()
    match = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", resolution)
    if match:
        return {
            "width": int(match.group(1)),
            "height": int(match.group(2)),
        }
    return resolution


class FalVideoService:
    def __init__(self) -> None:
        settings = get_settings()
        self.image_generation_model = settings.fal_image_model
        self.image_model = settings.fal_video_model
        self.text_model = settings.fal_text_video_model
        self.video_edit_model = settings.fal_video_edit_model
        self.api_key = settings.fal_key
        self.default_duration_seconds = settings.fal_video_duration_seconds
        os.environ["FAL_KEY"] = self.api_key

    async def _generate_single_clip(
        self, payload: VideoGenerationContext, clip_duration: str, clip_index: int = 0
    ) -> tuple[str, dict[str, Any]]:
        final_prompt = build_video_prompt(payload)
        if clip_index > 0:
            final_prompt += f" (Part {clip_index + 1} continuation)"

        arguments = {
            "prompt": final_prompt,
            "duration": clip_duration,
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
            raise RuntimeError(f"fal.AI did not return a video_url for clip duration {clip_duration}s.")

        provider_response = result if isinstance(result, dict) else {"result": result}
        return video_url, provider_response

    async def generate_video(
        self, payload: VideoGenerationContext
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("FAL_KEY is not configured.")
        if not payload.product_name:
            raise RuntimeError("product_name is required to generate a video.")
        if not payload.approved_script:
            raise RuntimeError("approved_script is required to generate a video.")

        total_seconds = payload.time_seconds or self.default_duration_seconds
        clip_durations = _calculate_clip_durations(total_seconds)
        final_prompt = build_video_prompt(payload)

        logger.info(
            f"Generating video for total duration {total_seconds}s split into clips: {clip_durations}"
        )

        generated_clips: list[str] = []
        last_provider_response: dict[str, Any] = {}

        for index, clip_dur in enumerate(clip_durations):
            video_url, resp = await self._generate_single_clip(
                payload=payload, clip_duration=clip_dur, clip_index=index
            )
            generated_clips.append(video_url)
            last_provider_response = resp

        if len(generated_clips) == 1:
            final_video_url = generated_clips[0]
        else:
            # Multi-clip duration (15s, 20s, 30s): Stitch with FFmpeg
            final_video_url = await asyncio.to_thread(self._stitch_video_clips, generated_clips)

        last_provider_response["total_duration_requested"] = total_seconds
        last_provider_response["clip_durations_generated"] = clip_durations
        last_provider_response["all_clip_urls"] = generated_clips

        return final_video_url, last_provider_response, final_prompt

    async def generate_image_from_prompt(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("FAL_KEY is not configured.")
        if not self.image_generation_model:
            raise RuntimeError("FAL_IMAGE_MODEL is not configured.")

        arguments = {
            "prompt": prompt,
            "image_size": _normalize_image_size(resolution),
            "aspect_ratio": aspect_ratio,
            "num_images": 1,
        }
        result = await asyncio.to_thread(
            fal_client.subscribe,
            self.image_generation_model,
            arguments=arguments,
            with_logs=True,
        )

        image_url = self._extract_media_url(result, preferred_keys=("image", "images"))
        if not image_url:
            raise RuntimeError("fal.AI did not return an image URL.")

        provider_response = result if isinstance(result, dict) else {"result": result}
        return image_url, provider_response, self.image_generation_model

    async def generate_video_from_prompt(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        time_seconds: int,
        audio: bool,
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("FAL_KEY is not configured.")

        arguments = {
            "prompt": prompt,
            "duration": str(time_seconds),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": audio,
        }
        result = await asyncio.to_thread(
            fal_client.subscribe,
            self.text_model,
            arguments=arguments,
            with_logs=True,
        )

        video_url = self._extract_video_url(result)
        if not video_url:
            raise RuntimeError("fal.AI did not return a video URL.")

        provider_response = result if isinstance(result, dict) else {"result": result}
        return video_url, provider_response, self.text_model

    async def edit_video_from_prompt(
        self,
        *,
        prompt: str,
        video_ref: str,
        image_ref: str | None = None,
        audio: bool = True,
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("FAL_KEY is not configured.")
        if not self.video_edit_model:
            raise RuntimeError("FAL_VIDEO_EDIT_MODEL is not configured.")
        if not video_ref:
            raise RuntimeError("video_ref is required for video editing.")

        arguments: dict[str, Any] = {
            "prompt": prompt,
            "video_url": video_ref,
            "audio": audio,
        }
        if image_ref:
            arguments["image_url"] = image_ref

        result = await asyncio.to_thread(
            fal_client.subscribe,
            self.video_edit_model,
            arguments=arguments,
            with_logs=True,
        )

        video_url = self._extract_video_url(result)
        if not video_url:
            raise RuntimeError("fal.AI did not return an edited video URL.")

        provider_response = result if isinstance(result, dict) else {"result": result}
        return video_url, provider_response, self.video_edit_model

    def _stitch_video_clips(self, clip_urls: list[str]) -> str:
        """
        Downloads video clips and uses FFmpeg to concatenate them into a single file.
        Returns the path to the concatenated MP4 file.
        """
        temp_dir = tempfile.mkdtemp(prefix="jctranz_video_")
        local_files: list[str] = []

        try:
            for idx, url in enumerate(clip_urls):
                file_path = os.path.join(temp_dir, f"clip_{idx}.mp4")
                urllib.request.urlretrieve(url, file_path)
                local_files.append(file_path)

            list_file_path = os.path.join(temp_dir, "clips.txt")
            with open(list_file_path, "w", encoding="utf-8") as f:
                for file_path in local_files:
                    # Escape backslashes for FFmpeg on Windows
                    clean_path = file_path.replace("\\", "/")
                    f.write(f"file '{clean_path}'\n")

            output_file_path = os.path.join(temp_dir, "stitched_final.mp4")

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file_path,
                "-c",
                "copy",
                output_file_path,
            ]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.exists(output_file_path):
                return output_file_path
            else:
                raise RuntimeError("FFmpeg stitching completed but output file missing.")
        except Exception as exc:
            logger.error(f"FFmpeg stitching failed: {exc}")
            # Fallback: return first clip if stitching fails
            return clip_urls[0]

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

    def _extract_media_url(
        self, result: Any, preferred_keys: tuple[str, ...] = ("image", "video", "output")
    ) -> str | None:
        if isinstance(result, str):
            if result.startswith(("http://", "https://", "data:")):
                return result
            return None

        if isinstance(result, list):
            for item in result:
                url = self._extract_media_url(item, preferred_keys=preferred_keys)
                if url:
                    return url
            return None

        if not isinstance(result, dict):
            return None

        for key in ("url", "image_url", "video_url", "file_url"):
            value = result.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://", "data:")):
                return value

        for key in preferred_keys + ("output", "data", "result", "media"):
            url = self._extract_media_url(result.get(key), preferred_keys=preferred_keys)
            if url:
                return url

        return None
