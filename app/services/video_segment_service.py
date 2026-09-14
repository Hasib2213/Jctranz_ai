from __future__ import annotations

import base64
import math
import mimetypes
import os
import shutil
import subprocess
import urllib.request
from typing import Any


class VideoSegmentService:
    max_direct_edit_seconds = 10.0
    provider_min_height = 720

    def prepare_source(self, source: str, temp_dir: str, filename: str = "source.mp4") -> str:
        destination = os.path.join(temp_dir, filename)
        if source.startswith(("http://", "https://")):
            urllib.request.urlretrieve(source, destination)
        else:
            shutil.copyfile(source, destination)

        if not os.path.exists(destination) or os.path.getsize(destination) == 0:
            raise RuntimeError("Video download produced an empty file.")
        return destination

    def probe_duration(self, video_path: str) -> float:
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
        result = self._run(cmd)
        duration = self._parse_duration(result.stdout.decode("utf-8", "replace").strip())
        if duration is not None:
            return duration

        stream_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=duration",
            "-of",
            "csv=p=0",
            video_path,
        ]
        stream_result = self._run(stream_cmd)
        durations = [
            parsed
            for line in stream_result.stdout.decode("utf-8", "replace").splitlines()
            if (parsed := self._parse_duration(line.strip())) is not None
        ]
        if durations:
            return max(durations)

        raise RuntimeError("Could not read video duration.")

    def probe_dimensions(self, video_path: str) -> tuple[int, int]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ]
        result = self._run(cmd)
        raw = result.stdout.decode("utf-8", "replace").strip()
        try:
            width_text, height_text = raw.split("x", 1)
            width = int(width_text)
            height = int(height_text)
        except (ValueError, AttributeError) as exc:
            raise RuntimeError("Could not read video dimensions.") from exc
        return self._even_dimension(width), self._even_dimension(height)

    def extract_analysis_frames(
        self,
        video_path: str,
        temp_dir: str,
        *,
        max_frames: int = 8,
    ) -> list[dict[str, Any]]:
        duration = self.probe_duration(video_path)
        if duration <= 0:
            return []

        frame_count = min(max_frames, max(1, int(duration)))
        if frame_count == 1:
            timestamps = [min(duration / 2, max(duration - 0.1, 0))]
        else:
            step = duration / frame_count
            timestamps = [
                min(index * step + step / 2, duration - 0.1)
                for index in range(frame_count)
            ]

        frames: list[dict[str, Any]] = []
        for index, timestamp in enumerate(timestamps):
            frame_path = os.path.join(temp_dir, f"analysis_frame_{index}.jpg")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                "scale=512:-1",
                "-q:v",
                "3",
                frame_path,
            ]
            self._run(cmd)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                frames.append(
                    {
                        "index": index,
                        "time": round(timestamp, 3),
                        "path": frame_path,
                        "data_url": self._file_to_data_url(frame_path),
                    }
                )
        return frames

    def split_for_segment_edit(
        self,
        video_path: str,
        temp_dir: str,
        *,
        start_time: float,
        end_time: float,
        preserve_audio: bool = False,
    ) -> dict[str, str | None]:
        duration = self.probe_duration(video_path)
        start_time = max(0.0, min(start_time, duration))
        end_time = max(start_time, min(end_time, duration))
        if end_time - start_time <= 0:
            raise RuntimeError("Detected edit segment has no duration.")

        before_path = None
        after_path = None
        target_path = os.path.join(temp_dir, "target_segment.mp4")

        if start_time > 0.05:
            before_path = os.path.join(temp_dir, "before_segment.mp4")
            self.extract_segment(
                video_path,
                before_path,
                start_time=0.0,
                end_time=start_time,
                preserve_audio=preserve_audio,
            )

        self.extract_segment(
            video_path,
            target_path,
            start_time=start_time,
            end_time=end_time,
            preserve_audio=preserve_audio,
        )

        if end_time < duration - 0.05:
            after_path = os.path.join(temp_dir, "after_segment.mp4")
            self.extract_segment(
                video_path,
                after_path,
                start_time=end_time,
                end_time=duration,
                preserve_audio=preserve_audio,
            )

        return {
            "before": before_path,
            "target": target_path,
            "after": after_path,
        }

    def split_into_chunks(
        self,
        video_path: str,
        temp_dir: str,
        *,
        max_seconds: float | None = None,
        preserve_audio: bool = False,
    ) -> list[dict[str, Any]]:
        duration = self.probe_duration(video_path)
        if duration <= 0:
            raise RuntimeError("Video has no readable duration.")

        max_seconds = max_seconds or self.max_direct_edit_seconds
        chunk_count = max(1, math.ceil(duration / max_seconds))
        chunk_seconds = duration / chunk_count
        chunks: list[dict[str, Any]] = []

        for index in range(chunk_count):
            start_time = index * chunk_seconds
            end_time = duration if index == chunk_count - 1 else (index + 1) * chunk_seconds
            chunk_path = os.path.join(temp_dir, f"global_chunk_{index}.mp4")
            self.extract_segment(
                video_path,
                chunk_path,
                start_time=start_time,
                end_time=end_time,
                preserve_audio=preserve_audio,
            )
            chunks.append(
                {
                    "index": index,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "duration_seconds": round(end_time - start_time, 3),
                    "path": chunk_path,
                }
            )

        return chunks

    def split_for_range_edits(
        self,
        video_path: str,
        temp_dir: str,
        *,
        ranges: list[dict[str, float]],
        preserve_audio: bool = False,
        max_edit_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        duration = self.probe_duration(video_path)
        edit_ranges = self._normalize_ranges(
            ranges,
            duration=duration,
            max_edit_seconds=max_edit_seconds or self.max_direct_edit_seconds,
        )
        if not edit_ranges:
            raise RuntimeError("No valid edit ranges were selected.")

        parts: list[dict[str, Any]] = []
        cursor = 0.0
        part_index = 0
        for edit_range in edit_ranges:
            start_time = edit_range["start_time"]
            end_time = edit_range["end_time"]
            if start_time > cursor + 0.05:
                part_index = self._append_timeline_part(
                    parts,
                    video_path,
                    temp_dir,
                    index=part_index,
                    mode="keep",
                    start_time=cursor,
                    end_time=start_time,
                    preserve_audio=preserve_audio,
                )

            part_index = self._append_timeline_part(
                parts,
                video_path,
                temp_dir,
                index=part_index,
                mode="edit",
                start_time=start_time,
                end_time=end_time,
                preserve_audio=preserve_audio,
            )
            cursor = end_time

        if cursor < duration - 0.05:
            self._append_timeline_part(
                parts,
                video_path,
                temp_dir,
                index=part_index,
                mode="keep",
                start_time=cursor,
                end_time=duration,
                preserve_audio=preserve_audio,
            )

        return parts

    def extract_segment(
        self,
        source_path: str,
        output_path: str,
        *,
        start_time: float,
        end_time: float,
        preserve_audio: bool = False,
    ) -> None:
        duration = max(end_time - start_time, 0.01)
        source_has_audio = preserve_audio and self._has_audio_stream(source_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            source_path,
        ]
        if preserve_audio and not source_has_audio:
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )
        cmd.extend(
            [
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
            ]
        )
        if preserve_audio:
            if source_has_audio:
                cmd.extend(["-map", "0:a:0?"])
            else:
                cmd.extend(["-map", "1:a:0"])
        else:
            cmd.append("-an")
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if preserve_audio:
            cmd.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2"])
            if not source_has_audio:
                cmd.append("-shortest")
        cmd.extend(
            [
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self._run(cmd)
        self._require_media_file(output_path, "FFmpeg segment extraction failed.")

    def merge_clips(
        self,
        clip_paths: list[str],
        temp_dir: str,
        *,
        width: int,
        height: int,
        preserve_audio: bool = False,
        audio_source_paths: list[str | None] | None = None,
    ) -> str:
        normalized_paths: list[str] = []
        for index, clip_path in enumerate(clip_paths):
            normalized_path = os.path.join(temp_dir, f"normalized_{index}.mp4")
            audio_source_path = (
                audio_source_paths[index]
                if audio_source_paths is not None and index < len(audio_source_paths)
                else None
            )
            self.normalize_video(
                clip_path,
                normalized_path,
                width=width,
                height=height,
                preserve_audio=preserve_audio,
                audio_source_path=audio_source_path,
            )
            normalized_paths.append(normalized_path)

        list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as file:
            for path in normalized_paths:
                file.write(f"file '{path.replace(os.sep, '/')}'\n")

        output_path = os.path.join(temp_dir, "merged_final.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
        ]
        self._run(cmd)
        self._require_media_file(output_path, "FFmpeg merge failed.")
        return output_path

    def normalize_video(
        self,
        source_path: str,
        output_path: str,
        *,
        width: int,
        height: int,
        preserve_audio: bool = False,
        audio_source_path: str | None = None,
    ) -> None:
        video_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24"
        )
        duration = self.probe_duration(source_path)
        source_has_audio = preserve_audio and self._has_audio_stream(source_path)
        fallback_has_audio = (
            preserve_audio
            and bool(audio_source_path)
            and self._has_audio_stream(audio_source_path)
        )
        use_fallback_input = (
            fallback_has_audio
            and os.path.abspath(audio_source_path or "") != os.path.abspath(source_path)
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
        ]
        audio_map = "0:a:0?"
        used_silent_audio = False
        if use_fallback_input:
            cmd.extend(["-i", audio_source_path or ""])
            audio_map = "1:a:0?"
        elif preserve_audio and fallback_has_audio:
            audio_map = "0:a:0?"
        elif preserve_audio and not source_has_audio:
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )
            audio_map = "1:a:0"
            used_silent_audio = True
        cmd.extend(
            [
                "-map",
                "0:v:0",
            ]
        )
        if preserve_audio:
            cmd.extend(["-map", audio_map])
        else:
            cmd.append("-an")
        cmd.extend(
            [
                "-t",
                f"{duration:.3f}",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if preserve_audio:
            cmd.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2"])
            if used_silent_audio:
                cmd.append("-shortest")
        cmd.extend(
            [
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self._run(cmd)
        self._require_media_file(output_path, "FFmpeg normalization failed.")

    def normalize_for_provider(
        self,
        source_path: str,
        output_path: str,
        *,
        preserve_audio: bool = False,
        min_height: int | None = None,
    ) -> dict[str, Any]:
        min_height = min_height or self.provider_min_height
        original_width, original_height = self.probe_dimensions(source_path)
        target_width, target_height = self._provider_safe_dimensions(
            original_width,
            original_height,
            min_height=min_height,
        )
        duration = self.probe_duration(source_path)
        source_has_audio = preserve_audio and self._has_audio_stream(source_path)
        video_filter = f"scale={target_width}:{target_height},setsar=1,fps=24"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            source_path,
        ]
        if preserve_audio and not source_has_audio:
            cmd.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )
        cmd.extend(
            [
                "-map",
                "0:v:0",
            ]
        )
        if preserve_audio:
            cmd.extend(["-map", "0:a:0?" if source_has_audio else "1:a:0"])
        else:
            cmd.append("-an")
        cmd.extend(
            [
                "-t",
                f"{duration:.3f}",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if preserve_audio:
            cmd.extend(["-c:a", "aac", "-ar", "48000", "-ac", "2"])
            if not source_has_audio:
                cmd.append("-shortest")
        cmd.extend(
            [
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self._run(cmd)
        self._require_media_file(output_path, "FFmpeg provider normalization failed.")
        return {
            "original_width": original_width,
            "original_height": original_height,
            "normalized_width": target_width,
            "normalized_height": target_height,
            "min_height": min_height,
            "duration_seconds": round(duration, 3),
            "preserve_audio": preserve_audio,
            "source_had_audio": source_has_audio,
        }

    def _file_to_data_url(self, file_path: str) -> str:
        mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _require_media_file(self, file_path: str, message: str) -> None:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise RuntimeError(message)

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return result

        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        output = stderr or stdout or "no FFmpeg output"
        raise RuntimeError(f"{cmd[0]} exited {result.returncode}: {output[-1200:]}")

    def _even_dimension(self, value: int) -> int:
        return max(2, value if value % 2 == 0 else value - 1)

    def _parse_duration(self, raw_duration: str) -> float | None:
        try:
            duration = float(raw_duration)
        except ValueError:
            return None
        if not math.isfinite(duration) or duration <= 0:
            return None
        return duration

    def _provider_safe_dimensions(
        self,
        width: int,
        height: int,
        *,
        min_height: int,
    ) -> tuple[int, int]:
        target_height = self._even_dimension(max(height, min_height))
        scale = target_height / max(height, 1)
        target_width = self._even_dimension(round(width * scale))
        return target_width, target_height

    def _normalize_ranges(
        self,
        ranges: list[dict[str, float]],
        *,
        duration: float,
        max_edit_seconds: float,
    ) -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []
        for value in ranges:
            try:
                start_time = float(value.get("start_time", 0.0))
                end_time = float(value.get("end_time", 0.0))
            except (TypeError, ValueError, AttributeError):
                continue

            start_time = max(0.0, min(start_time, duration))
            end_time = max(start_time, min(end_time, duration))
            if end_time - start_time < 0.25:
                continue

            candidates.append(
                {
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                }
            )

        candidates.sort(key=lambda item: item["start_time"])
        merged: list[dict[str, float]] = []
        for value in candidates:
            if not merged or value["start_time"] > merged[-1]["end_time"] + 0.05:
                merged.append(dict(value))
                continue
            merged[-1]["end_time"] = max(merged[-1]["end_time"], value["end_time"])

        normalized: list[dict[str, float]] = []
        for value in merged:
            start_time = value["start_time"]
            end_time = value["end_time"]
            while end_time - start_time > max_edit_seconds:
                split_end = min(end_time, start_time + max_edit_seconds)
                normalized.append(
                    {
                        "start_time": round(start_time, 3),
                        "end_time": round(split_end, 3),
                    }
                )
                start_time = split_end

            if end_time - start_time >= 0.25:
                normalized.append(
                    {
                        "start_time": round(start_time, 3),
                        "end_time": round(end_time, 3),
                    }
                )

        return normalized

    def _append_timeline_part(
        self,
        parts: list[dict[str, Any]],
        video_path: str,
        temp_dir: str,
        *,
        index: int,
        mode: str,
        start_time: float,
        end_time: float,
        preserve_audio: bool,
    ) -> int:
        output_path = os.path.join(temp_dir, f"timeline_{index}_{mode}.mp4")
        self.extract_segment(
            video_path,
            output_path,
            start_time=start_time,
            end_time=end_time,
            preserve_audio=preserve_audio,
        )
        parts.append(
            {
                "index": index,
                "mode": mode,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration_seconds": round(end_time - start_time, 3),
                "path": output_path,
            }
        )
        return index + 1

    def _has_audio_stream(self, video_path: str) -> bool:
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return False

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            video_path,
        ]
        try:
            result = self._run(cmd)
        except RuntimeError:
            return False
        return bool(result.stdout.decode("utf-8", "replace").strip())
