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
            "motion, lighting, camera direction, and constraints where useful."
        )
        if generation_type == "video_edit":
            instruction = (
                "Rewrite the user's video edit instruction into a precise production prompt. "
                "Strictly preserve the requested edit and do not add unrelated changes."
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

    async def generate_promotional_script(self, payload: ScriptGenerationRequest) -> dict[str, Any]:
        prompt = build_script_prompt(payload)
        return await self.generate_text_from_prompt(prompt)
