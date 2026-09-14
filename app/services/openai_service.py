import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.ai_models import ScriptGenerationRequest
from app.prompts.script_prompt import build_script_prompt
from app.services.cost_service import CostService


class OpenAIScriptService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_script_model
        self.prompt_model = settings.openai_prompt_model
        self.api_key = settings.openai_api_key
        self.costs = CostService()

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

        return json.loads(cleaned_text)

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)

        if not texts:
            raise RuntimeError("OpenAI did not return text.")
        return "\n".join(texts).strip()

    async def _create_text_response(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 350,
    ) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        client = AsyncOpenAI(api_key=self.api_key)
        return await client.responses.create(
            model=model or self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def generate_text_from_prompt(self, prompt: str) -> dict[str, Any]:
        response = await self._create_text_response(prompt)
        return self._parse_json_response(self._extract_response_text(response))

    def _build_refine_prompt(
        self,
        *,
        user_prompt: str,
        generation_type: str,
        settings: dict[str, Any] | None = None,
        references: dict[str, Any] | None = None,
        previous_refined_prompt: str | None = None,
    ) -> str:
        settings_text = json.dumps(settings or {}, ensure_ascii=False)
        references_text = json.dumps(references or {}, ensure_ascii=False)

        instruction = (
            "Rewrite the user's prompt into a high-quality production prompt for the target AI "
            "media model. Keep the user's intent. Improve clarity, visual detail, composition, "
            "motion, lighting, camera direction, and constraints where useful. Keep the prompt "
            "provider-safe: avoid graphic violence, sexual content, hate, weapons, drugs, "
            "celebrity likenesses, copyrighted characters, brand logos, readable text, captions, "
            "and watermark requests unless the user explicitly owns or requires them. If people "
            "appear, make them generic and non-identifiable unless the user provided a valid "
            "reference."
        )
        if generation_type == "video_edit":
            instruction = (
                "Rewrite the user's video edit instruction into a precise production prompt. "
                "Strictly preserve the requested edit and do not add unrelated changes. Keep it "
                "provider-safe, non-graphic, non-sexual, and free of real-person impersonation. "
                "Do not request logos, captions, watermarks, or unrelated new content."
            )

        previous_block = ""
        if previous_refined_prompt:
            previous_block = (
                "\nPrevious refined prompt:\n"
                f"{previous_refined_prompt}\n"
                "Create a better version while preserving the user's original intent.\n"
            )

        return (
            f"{instruction}\n\n"
            f"Generation type: {generation_type}\n"
            f"Settings: {settings_text}\n"
            f"References: {references_text}\n"
            f"{previous_block}\n"
            f"User prompt:\n{user_prompt}\n\n"
            "Return only valid JSON in this exact shape:\n"
            '{"refined_prompt":"..."}'
        )

    async def refine_generation_prompt(
        self,
        *,
        user_prompt: str,
        generation_type: str,
        settings: dict[str, Any] | None = None,
        references: dict[str, Any] | None = None,
        previous_refined_prompt: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_refine_prompt(
            user_prompt=user_prompt,
            generation_type=generation_type,
            settings=settings,
            references=references,
            previous_refined_prompt=previous_refined_prompt,
        )
        response = await self._create_text_response(
            prompt,
            model=self.prompt_model,
            temperature=0.5,
            max_output_tokens=700,
        )
        text = self._extract_response_text(response)
        try:
            parsed = self._parse_json_response(text)
        except json.JSONDecodeError:
            parsed = {"refined_prompt": text.strip()}

        refined_prompt = parsed.get("refined_prompt")
        if not isinstance(refined_prompt, str) or not refined_prompt.strip():
            raise RuntimeError("OpenAI did not return a valid refined_prompt.")

        cost_detail = self.costs.openai_cost_detail(response)
        return {
            "refined_prompt": refined_prompt.strip(),
            "usage": {
                "input_tokens": cost_detail["input_tokens"],
                "output_tokens": cost_detail["output_tokens"],
                "total_tokens": cost_detail["total_tokens"],
            },
            "cost": cost_detail["cost"],
            "formatted_cost": cost_detail["formatted_cost"],
            "model": self.prompt_model,
        }

    def _build_media_plan_prompt(
        self,
        *,
        user_prompt: str,
        generation_type: str,
        settings: dict[str, Any] | None = None,
        references: dict[str, Any] | None = None,
    ) -> str:
        settings_text = json.dumps(settings or {}, ensure_ascii=False)
        references_text = json.dumps(references or {}, ensure_ascii=False)
        return (
            "You are the planner for an AI media generation backend. Create a compact, "
            "provider-safe execution plan for the Python executor. Do not call tools. Do not "
            "invent unsupported API fields. Prefer Magica for image, video, and video edit when "
            "it fits; allow fal_ai as fallback. Keep user intent, but reduce moderation risk.\n\n"
            f"Generation type: {generation_type}\n"
            f"Settings: {settings_text}\n"
            f"References: {references_text}\n"
            f"User prompt:\n{user_prompt}\n\n"
            "Return only valid JSON in this exact shape:\n"
            "{"
            '"task_type":"image_generation|video_generation|video_edit",'
            '"preferred_provider":"magica|fal_ai",'
            '"risk_level":"low|medium|high",'
            '"prompt_guidance":"short provider-safe guidance for prompt refinement",'
            '"retry_strategy":["original","safe_prompt","fallback_provider"],'
            '"settings_overrides":{}'
            "}"
        )

    async def plan_media_generation(
        self,
        *,
        user_prompt: str,
        generation_type: str,
        settings: dict[str, Any] | None = None,
        references: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = self._build_media_plan_prompt(
            user_prompt=user_prompt,
            generation_type=generation_type,
            settings=settings,
            references=references,
        )
        response = await self._create_text_response(
            prompt,
            model=self.prompt_model,
            temperature=0.2,
            max_output_tokens=450,
        )
        text = self._extract_response_text(response)
        try:
            parsed = self._parse_json_response(text)
        except json.JSONDecodeError:
            parsed = {}

        preferred_provider = parsed.get("preferred_provider")
        if preferred_provider not in {"magica", "fal_ai"}:
            preferred_provider = "magica"

        task_type = parsed.get("task_type")
        allowed_task_types = {
            "image_generation",
            "video_generation",
            "video_edit",
        }
        if task_type not in allowed_task_types:
            task_type = generation_type

        risk_level = parsed.get("risk_level")
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"

        retry_strategy = parsed.get("retry_strategy")
        if not isinstance(retry_strategy, list):
            retry_strategy = ["original", "safe_prompt", "fallback_provider"]

        settings_overrides = parsed.get("settings_overrides")
        if not isinstance(settings_overrides, dict):
            settings_overrides = {}

        prompt_guidance = parsed.get("prompt_guidance")
        if not isinstance(prompt_guidance, str):
            prompt_guidance = ""

        cost_detail = self.costs.openai_cost_detail(response)
        return {
            "task_type": task_type,
            "preferred_provider": preferred_provider,
            "risk_level": risk_level,
            "prompt_guidance": prompt_guidance.strip(),
            "retry_strategy": [str(item) for item in retry_strategy[:5]],
            "settings_overrides": settings_overrides,
            "usage": {
                "input_tokens": cost_detail["input_tokens"],
                "output_tokens": cost_detail["output_tokens"],
                "total_tokens": cost_detail["total_tokens"],
            },
            "cost": cost_detail["cost"],
            "formatted_cost": cost_detail["formatted_cost"],
            "model": self.prompt_model,
        }

    def _build_video_segment_analysis_text(
        self,
        *,
        user_prompt: str,
        refined_prompt: str,
        duration_seconds: float,
        frame_times: list[float],
        max_segment_seconds: float,
    ) -> str:
        frame_times_text = ", ".join(f"{time_value:.2f}s" for time_value in frame_times)
        return (
            "You are a video edit planner. Analyze the provided sampled frames and decide which "
            "time range or ranges should be edited. The Python executor can leave unchanged "
            "parts untouched, edit selected ranges, or edit all chunks when the target appears "
            "throughout the video. Do not ask the user for clarification; "
            "make the best reasonable choice even if confidence is low.\n\n"
            f"Original user instruction: {user_prompt}\n"
            f"Refined edit prompt: {refined_prompt}\n"
            f"Video duration: {duration_seconds:.2f}s\n"
            f"Sampled frame timestamps in order: {frame_times_text}\n"
            f"Maximum target segment duration: {max_segment_seconds:.2f}s\n\n"
            "Return only valid JSON in this exact shape:\n"
            "{"
            '"requires_segment_edit":true,'
            '"target_object":"object/action to edit or empty string",'
            '"replacement_object":"replacement object/style or empty string",'
            '"start_time":0.0,'
            '"end_time":0.0,'
            '"coverage":"localized|partial|throughout|unclear",'
            '"target_ranges":[{"start_time":0.0,"end_time":0.0}],'
            '"confidence":"low|medium|high",'
            '"reason":"brief reason based on visible frames",'
            '"edit_prompt":"precise prompt for editing only the selected clip"'
            "}\n\n"
            "Rules: choose ranges where the target object/action is visible. Use coverage "
            "\"throughout\" only when the target appears in nearly all sampled frames; use "
            "\"localized\" when it appears in one compact time span; use \"partial\" for multiple "
            "or broad but not full-video spans; use \"unclear\" when uncertain. target_ranges "
            "should cover only likely visible target spans and may be longer than the max segment "
            "duration because the executor can split them later. start_time/end_time should be "
            "the best single range fallback. Keep all times within video duration."
        )

    async def analyze_video_edit_segment(
        self,
        *,
        user_prompt: str,
        refined_prompt: str,
        duration_seconds: float,
        frames: list[dict[str, Any]],
        max_segment_seconds: float,
    ) -> dict[str, Any]:
        if not frames:
            raise RuntimeError("No video frames were available for analysis.")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        frame_times = [float(frame.get("time") or 0.0) for frame in frames]
        text = self._build_video_segment_analysis_text(
            user_prompt=user_prompt,
            refined_prompt=refined_prompt,
            duration_seconds=duration_seconds,
            frame_times=frame_times,
            max_segment_seconds=max_segment_seconds,
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for frame in frames:
            data_url = frame.get("data_url")
            if isinstance(data_url, str) and data_url.startswith("data:image/"):
                content.append({"type": "input_image", "image_url": data_url})

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.responses.create(
            model=self.prompt_model,
            input=[{"role": "user", "content": content}],
            temperature=0.1,
            max_output_tokens=650,
        )
        output_text = self._extract_response_text(response)
        try:
            parsed = self._parse_json_response(output_text)
        except json.JSONDecodeError:
            parsed = {}

        start_time = self._coerce_float(parsed.get("start_time"), 0.0)
        end_time = self._coerce_float(
            parsed.get("end_time"),
            min(duration_seconds, max_segment_seconds),
        )
        start_time = max(0.0, min(start_time, duration_seconds))
        end_time = max(start_time, min(end_time, duration_seconds))
        if end_time - start_time > max_segment_seconds:
            end_time = min(duration_seconds, start_time + max_segment_seconds)
        if end_time - start_time < 0.5:
            end_time = min(
                duration_seconds,
                start_time + min(max_segment_seconds, duration_seconds),
            )
        target_ranges = self._coerce_time_ranges(
            parsed.get("target_ranges"),
            duration_seconds=duration_seconds,
            fallback_start=start_time,
            fallback_end=end_time,
        )

        coverage = parsed.get("coverage")
        if coverage not in {"localized", "partial", "throughout", "unclear"}:
            coverage = "unclear"

        confidence = parsed.get("confidence")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"

        cost_detail = self.costs.openai_cost_detail(response)
        return {
            "requires_segment_edit": True,
            "target_object": str(parsed.get("target_object") or "").strip(),
            "replacement_object": str(parsed.get("replacement_object") or "").strip(),
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "coverage": coverage,
            "target_ranges": target_ranges,
            "confidence": confidence,
            "reason": str(parsed.get("reason") or "").strip(),
            "edit_prompt": str(parsed.get("edit_prompt") or refined_prompt).strip(),
            "usage": {
                "input_tokens": cost_detail["input_tokens"],
                "output_tokens": cost_detail["output_tokens"],
                "total_tokens": cost_detail["total_tokens"],
            },
            "cost": cost_detail["cost"],
            "formatted_cost": cost_detail["formatted_cost"],
            "model": self.prompt_model,
        }

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_time_ranges(
        self,
        value: Any,
        *,
        duration_seconds: float,
        fallback_start: float,
        fallback_end: float,
    ) -> list[dict[str, float]]:
        if not isinstance(value, list):
            value = [{"start_time": fallback_start, "end_time": fallback_end}]

        ranges: list[dict[str, float]] = []
        for item in value:
            if isinstance(item, dict):
                start_time = self._coerce_float(item.get("start_time"), fallback_start)
                end_time = self._coerce_float(item.get("end_time"), fallback_end)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start_time = self._coerce_float(item[0], fallback_start)
                end_time = self._coerce_float(item[1], fallback_end)
            else:
                continue

            start_time = max(0.0, min(start_time, duration_seconds))
            end_time = max(start_time, min(end_time, duration_seconds))
            if end_time - start_time < 0.25:
                continue
            ranges.append(
                {
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                }
            )

        if not ranges and fallback_end > fallback_start:
            ranges.append(
                {
                    "start_time": round(fallback_start, 3),
                    "end_time": round(fallback_end, 3),
                }
            )
        return ranges

    async def generate_promotional_script(self, payload: ScriptGenerationRequest) -> dict[str, Any]:
        prompt = build_script_prompt(payload)
        return await self.generate_text_from_prompt(prompt)
