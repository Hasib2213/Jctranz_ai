from typing import Any
from app.adapters.base_adapter import BaseAIAdapter
from app.core.errors import (
    GenerationFailedException,
    ProviderTimeoutException,
    ProviderUnavailableException,
    RateLimitException,
)
from app.services.fal_service import FalVideoService


class FalVideoAdapter(BaseAIAdapter):
    def __init__(self) -> None:
        self.service = FalVideoService()

    @property
    def provider_name(self) -> str:
        return "fal_ai"

    @property
    def capability(self) -> str:
        return "video"

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        context_payload = payload.get("context_payload")
        if not context_payload:
            raise GenerationFailedException("Missing context_payload for FalVideoAdapter.")

        try:
            video_url, provider_response, final_prompt = await self.service.generate_video(
                context_payload
            )
            return {
                "success": True,
                "video_url": video_url,
                "provider_response": provider_response,
                "final_prompt": final_prompt,
                "provider": self.provider_name,
            }
        except Exception as exc:
            err_str = str(exc).lower()
            if "timeout" in err_str:
                raise ProviderTimeoutException(f"Fal.ai timeout: {exc}") from exc
            elif "rate limit" in err_str or "429" in err_str:
                raise RateLimitException(f"Fal.ai rate limit: {exc}") from exc
            elif "unavailable" in err_str or "503" in err_str:
                raise ProviderUnavailableException(f"Fal.ai service unavailable: {exc}") from exc
            else:
                raise GenerationFailedException(f"Fal.ai generation failed: {exc}") from exc
