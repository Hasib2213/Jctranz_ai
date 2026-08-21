import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.ai_models import ScriptGenerationRequest
from app.prompts.script_prompt import build_script_prompt


class OpenAIScriptService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_script_model
        self.api_key = settings.openai_api_key

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

        return json.loads(cleaned_text)

    async def generate_text_from_prompt(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.7,
            max_output_tokens=350,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return self._parse_json_response(output_text)

        # Fallback for SDK response shapes that expose text inside output items.
        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)

        if not texts:
            raise RuntimeError("OpenAI did not return a script.")
        return self._parse_json_response("\n".join(texts).strip())

    async def generate_promotional_script(self, payload: ScriptGenerationRequest) -> dict[str, Any]:
        prompt = build_script_prompt(payload)
        return await self.generate_text_from_prompt(prompt)
