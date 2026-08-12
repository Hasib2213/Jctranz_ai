from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.ai_models import ScriptGenerationRequest
from app.prompts.script_prompt import build_script_prompt


class OpenAIScriptService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.openai_script_model
        self.api_key = settings.openai_api_key

    async def generate_promotional_script(self, payload: ScriptGenerationRequest) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        client = AsyncOpenAI(api_key=self.api_key)
        prompt = build_script_prompt(payload)
        response = await client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.7,
            max_output_tokens=350,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        # Fallback for SDK response shapes that expose text inside output items.
        texts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)

        if not texts:
            raise RuntimeError("OpenAI did not return a script.")
        return "\n".join(texts).strip()
