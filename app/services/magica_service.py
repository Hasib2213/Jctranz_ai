from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

from app.core.config import get_settings
from app.services.cloudinary_service import CloudinaryService
from app.services.fal_service import _build_text_video_segment_prompt

logger = logging.getLogger(__name__)


def _calculate_magica_clip_durations(total_seconds: int) -> list[str]:
    if total_seconds <= 15:
        return [str(total_seconds)]
    if total_seconds == 20:
        return ["15", "5"]
    if total_seconds == 30:
        return ["15", "15"]
    raise RuntimeError("Magica video duration must be 5, 10, 15, 20, or 30 seconds.")


class MagicaVideoService:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.magica_api_key
        self.base_url = settings.magica_base_url.rstrip("/")
        self.image_node_type = settings.magica_image_model_node_type
        self.text_image_submodel = settings.magica_text_image_submodel
        self.image_output_format = settings.magica_image_output_format
        self.node_type = settings.magica_video_model_node_type
        self.text_submodel = settings.magica_text_video_submodel
        self.image_submodel = settings.magica_image_video_submodel
        self.video_edit_node_type = settings.magica_video_edit_model_node_type
        self.poll_interval_seconds = settings.magica_poll_interval_seconds
        self.timeout_seconds = settings.magica_timeout_seconds
        self.storage = CloudinaryService()
        self.prices_by_resolution = {
            "480p": settings.magica_480p_price_per_second_usd,
            "720p": settings.magica_720p_price_per_second_usd,
        }

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def generate_image_from_prompt(
        self,
        *,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("MAGICA_API_KEY is not configured.")
        if not self.image_node_type:
            raise RuntimeError("MAGICA_IMAGE_MODEL_NODE_TYPE is not configured.")

        input_payload = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": self._magica_image_size(resolution, aspect_ratio),
            "output_format": self.image_output_format or "JPEG",
        }
        run = await self._run_and_wait(
            node_type=self.image_node_type,
            submodel_id=self.text_image_submodel or None,
            input_payload=input_payload,
        )
        image_url = self._extract_media_url(run.get("output"))
        if not image_url:
            raise RuntimeError(f"Magica did not return an image URL for run {run.get('id')}.")

        provider_response = {
            "provider": "magica",
            "node_type": self.image_node_type,
            "submodel": self.text_image_submodel,
            "run_id": run.get("id"),
            "status": run.get("status"),
            "creditUsed": run.get("creditUsed"),
            "image_url": image_url,
        }
        return image_url, provider_response, self.text_image_submodel or self.image_node_type

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
            raise RuntimeError("MAGICA_API_KEY is not configured.")

        clip_durations = _calculate_magica_clip_durations(time_seconds)
        generated_clips: list[str] = []
        clip_responses: list[dict[str, Any]] = []
        continuation_frame_url: str | None = None

        with tempfile.TemporaryDirectory(prefix="jctranz_magica_continuity_") as continuity_dir:
            for index, clip_duration in enumerate(clip_durations):
                uses_reference_frame = continuation_frame_url is not None
                segment_prompt = _build_text_video_segment_prompt(
                    prompt=prompt,
                    clip_index=index,
                    clip_count=len(clip_durations),
                    clip_duration=clip_duration,
                    total_seconds=time_seconds,
                    uses_start_image=uses_reference_frame,
                )

                if uses_reference_frame:
                    input_payload = {
                        "image_url": continuation_frame_url,
                        "prompt": segment_prompt,
                        "duration": clip_duration,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "generate_audio": audio,
                    }
                    submodel = self.image_submodel
                else:
                    input_payload = {
                        "prompt": segment_prompt,
                        "duration": clip_duration,
                        "aspect_ratio": aspect_ratio,
                        "resolution": resolution,
                        "generate_audio": audio,
                    }
                    submodel = self.text_submodel

                run = await self._run_and_wait(
                    node_type=self.node_type,
                    submodel_id=submodel,
                    input_payload=input_payload,
                )
                video_url = self._extract_media_url(run.get("output"))
                if not video_url:
                    raise RuntimeError(
                        f"Magica did not return a video URL for run {run.get('id')}."
                    )

                generated_clips.append(video_url)
                clip_responses.append(
                    {
                        "run_id": run.get("id"),
                        "status": run.get("status"),
                        "subModelId": run.get("subModelId"),
                        "creditUsed": run.get("creditUsed"),
                        "video_url": video_url,
                        "duration": clip_duration,
                        "used_reference_frame": uses_reference_frame,
                    }
                )

                if index < len(clip_durations) - 1:
                    continuation_frame_url = await asyncio.to_thread(
                        self._extract_and_upload_last_frame,
                        video_url,
                        continuity_dir,
                        index,
                    )

        if len(generated_clips) == 1:
            final_video_url = generated_clips[0]
        else:
            final_video_url = await asyncio.to_thread(self._stitch_video_clips, generated_clips)

        provider_response = {
            "provider": "magica",
            "node_type": self.node_type,
            "text_submodel": self.text_submodel,
            "image_submodel": self.image_submodel,
            "total_duration_requested": time_seconds,
            "clip_durations_generated": clip_durations,
            "all_clip_urls": generated_clips,
            "clip_count": len(generated_clips),
            "generate_audio": audio,
            "continuation_mode": (
                "last_frame_image_to_video" if len(generated_clips) > 1 else "none"
            ),
            "estimated_cost_usd": self._estimate_cost(time_seconds, resolution),
            "clip_responses": clip_responses,
        }
        model = (
            self.text_submodel
            if len(generated_clips) == 1
            else f"{self.text_submodel}+{self.image_submodel}"
        )
        return final_video_url, provider_response, model

    async def edit_video_from_prompt(
        self,
        *,
        prompt: str,
        video_ref: str,
        image_ref: str | None = None,
        audio: bool = True,
    ) -> tuple[str, dict[str, Any], str]:
        if not self.api_key:
            raise RuntimeError("MAGICA_API_KEY is not configured.")
        if not self.video_edit_node_type:
            raise RuntimeError("MAGICA_VIDEO_EDIT_MODEL_NODE_TYPE is not configured.")
        if not video_ref:
            raise RuntimeError("video_ref is required for video editing.")

        input_payload: dict[str, Any] = {
            "video_url": video_ref,
            "prompt": prompt,
            "keep_audio": audio,
        }
        if image_ref:
            input_payload["image_urls"] = [image_ref]

        run = await self._run_and_wait(
            node_type=self.video_edit_node_type,
            submodel_id=None,
            input_payload=input_payload,
        )
        video_url = self._extract_media_url(run.get("output"))
        if not video_url:
            raise RuntimeError(
                f"Magica did not return an edited video URL for run {run.get('id')}."
            )

        provider_response = {
            "provider": "magica",
            "node_type": self.video_edit_node_type,
            "run_id": run.get("id"),
            "status": run.get("status"),
            "creditUsed": run.get("creditUsed"),
            "video_url": video_url,
            "keep_audio": audio,
        }
        return video_url, provider_response, self.video_edit_node_type

    async def _run_and_wait(
        self,
        *,
        node_type: str,
        submodel_id: str | None,
        input_payload: dict[str, Any],
    ) -> dict[str, Any]:
        run_body: dict[str, Any] = {"input": input_payload}
        if submodel_id:
            run_body["subModelId"] = submodel_id

        run_id = await asyncio.to_thread(
            self._start_run,
            node_type,
            run_body,
        )

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            run = await asyncio.to_thread(self._get_run, run_id)
            status = str(run.get("status", "")).upper()
            if status == "COMPLETED":
                return run
            if status in {"FAILED", "CANCELED"}:
                error = run.get("error") or run.get("userMessage") or "unknown error"
                raise RuntimeError(f"Magica run {run_id} ended with {status}: {error}")
            await asyncio.sleep(self.poll_interval_seconds)

        raise RuntimeError(f"Magica run {run_id} did not finish within {self.timeout_seconds}s.")

    def _start_run(self, node_type: str, body: dict[str, Any]) -> str:
        response = self._request_json(
            method="POST",
            url=f"{self.base_url}/nodes/{node_type}/run",
            body=body,
        )
        run_id = response.get("runId")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Magica did not return a runId.")
        return run_id

    def _get_run(self, run_id: str) -> dict[str, Any]:
        return self._request_json(
            method="GET",
            url=f"{self.base_url}/nodes/runs/{run_id}",
        )

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "JctranzAI/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Magica API error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Magica API connection failed: {exc}") from exc

    def _extract_and_upload_last_frame(
        self, video_url: str, temp_dir: str, clip_index: int
    ) -> str:
        if not self.storage.is_configured:
            raise RuntimeError("Cloudinary is required for Magica multi-clip continuation.")

        local_video_path = os.path.join(temp_dir, f"continuity_clip_{clip_index}.mp4")
        frame_path = os.path.join(temp_dir, f"continuity_frame_{clip_index}.jpg")

        self._download_or_copy_media(video_url, local_video_path)
        self._extract_continuation_frame(local_video_path, frame_path)

        upload = self.storage.upload_image(
            frame_path,
            job_id=f"magica_continuity_{clip_index}_{int(time.time())}",
        )
        frame_url = upload.get("secure_url")
        if not frame_url:
            raise RuntimeError("Cloudinary did not return a continuation frame URL.")
        return frame_url

    def _download_or_copy_media(self, source: str, destination: str) -> None:
        try:
            if source.startswith(("http://", "https://")):
                urllib.request.urlretrieve(source, destination)
            else:
                shutil.copyfile(source, destination)
        except Exception as exc:
            raise RuntimeError(f"Could not prepare Magica video clip for FFmpeg: {exc}") from exc

        if not os.path.exists(destination) or os.path.getsize(destination) == 0:
            raise RuntimeError("Magica video clip download produced an empty file.")

    def _extract_continuation_frame(self, video_path: str, frame_path: str) -> None:
        attempts: list[list[str]] = []
        duration = self._probe_video_duration(video_path)
        if duration is not None:
            attempts.append(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(max(duration - 0.2, 0)),
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_path,
                ]
            )

        attempts.extend(
            [
                [
                    "ffmpeg",
                    "-y",
                    "-sseof",
                    "-0.5",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_path,
                ],
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vf",
                    r"select=eq(n\,0)",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_path,
                ],
            ]
        )

        errors: list[str] = []
        for cmd in attempts:
            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if (
                result.returncode == 0
                and os.path.exists(frame_path)
                and os.path.getsize(frame_path) > 0
            ):
                return

            if os.path.exists(frame_path):
                try:
                    os.remove(frame_path)
                except OSError:
                    pass
            errors.append(self._format_process_error(cmd, result))

        raise RuntimeError(
            "FFmpeg did not extract a Magica continuation frame. "
            f"Attempts failed: {' | '.join(errors)}"
        )

    def _probe_video_duration(self, video_path: str) -> float | None:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            logger.warning(
                "Magica FFprobe duration failed: %s",
                result.stderr.decode("utf-8", "replace"),
            )
            return None

        try:
            duration = float(result.stdout.decode("utf-8", "replace").strip())
        except ValueError:
            return None
        return duration if duration > 0 else None

    def _format_process_error(
        self, cmd: list[str], result: subprocess.CompletedProcess[bytes]
    ) -> str:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        output = stderr or stdout or "no FFmpeg output"
        return f"{cmd[0]} exited {result.returncode}: {output[-1000:]}"

    def _stitch_video_clips(self, clip_urls: list[str]) -> str:
        temp_dir = tempfile.mkdtemp(prefix="jctranz_magica_video_")
        local_files: list[str] = []

        try:
            for idx, url in enumerate(clip_urls):
                file_path = os.path.join(temp_dir, f"clip_{idx}.mp4")
                urllib.request.urlretrieve(url, file_path)
                local_files.append(file_path)

            list_file_path = os.path.join(temp_dir, "clips.txt")
            with open(list_file_path, "w", encoding="utf-8") as file:
                for file_path in local_files:
                    clean_path = file_path.replace("\\", "/")
                    file.write(f"file '{clean_path}'\n")

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
            raise RuntimeError("FFmpeg stitching completed but output file missing.")
        except Exception as exc:
            logger.error("Magica FFmpeg stitching failed: %s", exc)
            raise RuntimeError(f"Magica FFmpeg stitching failed: {exc}") from exc

    def _estimate_cost(self, time_seconds: int, resolution: str) -> float:
        price = self.prices_by_resolution.get(resolution.lower(), 0.0)
        return round(time_seconds * price, 8)

    def _magica_image_size(self, resolution: str, aspect_ratio: str) -> str | dict[str, int]:
        clean_aspect_ratio = aspect_ratio.strip()
        if clean_aspect_ratio in {"1:1", "4:3", "3:4", "16:9", "9:16"}:
            return clean_aspect_ratio

        clean_resolution = resolution.strip().lower().replace(" ", "")
        resolution_map = {
            "480p": {"width": 854, "height": 480},
            "720p": {"width": 1280, "height": 720},
            "1080p": {"width": 1920, "height": 1080},
        }
        if clean_resolution in resolution_map:
            return resolution_map[clean_resolution]
        return "16:9"

    def _extract_media_url(self, value: Any) -> str | None:
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                return value
            return None
        if isinstance(value, list):
            for item in value:
                url = self._extract_media_url(item)
                if url:
                    return url
            return None
        if not isinstance(value, dict):
            return None

        for key in ("video_url", "url", "file_url", "content_url"):
            url = value.get(key)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return url

        for key in ("video", "videos", "output", "data", "result", "media"):
            url = self._extract_media_url(value.get(key))
            if url:
                return url
        return None
