from typing import Any
import openai
from app.adapters.base_adapter import BaseAIAdapter
from app.core.errors import (
    GenerationFailedException,
    ProviderTimeoutException,
    ProviderUnavailableException,
    RateLimitException,
)
from app.services.openai_service import OpenAIScriptService


class OpenAIAdapter(BaseAIAdapter):
    def __init__(self) -> None:
        self.service = OpenAIScriptService()

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def capability(self) -> str:
        return "script"

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_type = payload.get("task_type", "generate_script")
        try:
            if task_type == "generate_script":
                req_payload = payload.get("request_payload")
                result = await self.service.generate_promotional_script(req_payload)
                return {"success": True, "result": result, "provider": self.provider_name}
            elif task_type == "custom_prompt":
                prompt = payload.get("prompt", "")
                result = await self.service.generate_text_from_prompt(prompt)
                return {"success": True, "result": result, "provider": self.provider_name}
            else:
                raise GenerationFailedException(f"Unknown OpenAI task type: {task_type}")
        except openai.RateLimitError as exc:
            raise RateLimitException(f"OpenAI rate limit hit: {exc}") from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutException(f"OpenAI API timed out: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailableException(f"OpenAI connection error: {exc}") from exc
        except Exception as exc:
            raise GenerationFailedException(f"OpenAI execution error: {exc}") from exc
