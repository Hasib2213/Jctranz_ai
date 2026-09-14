import logging
from typing import Any
from app.adapters.base_adapter import BaseAIAdapter
from app.adapters.fal_adapter import FalVideoAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.core.errors import (
    AIOrchestratorException,
    ProviderTimeoutException,
    ProviderUnavailableException,
    RateLimitException,
)

logger = logging.getLogger(__name__)


class RoutingService:
    def __init__(self) -> None:
        self.adapters: dict[str, list[BaseAIAdapter]] = {
            "script": [OpenAIAdapter()],
            "video": [FalVideoAdapter()],
        }

    def register_adapter(self, capability: str, adapter: BaseAIAdapter) -> None:
        if capability not in self.adapters:
            self.adapters[capability] = []
        self.adapters[capability].append(adapter)

    async def execute_with_routing_and_failover(
        self, capability: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Executes request using primary provider adapter for capability.
        If primary fails due to Timeout/RateLimit/Unavailable, automatically fails over to backup adapter.
        """
        adapters_list = self.adapters.get(capability, [])
        if not adapters_list:
            raise AIOrchestratorException(
                error_code="AI_INVALID_REQUEST",
                message=f"No provider adapter registered for capability: {capability}",
                status_code=400,
            )

        last_exception = None
        for index, adapter in enumerate(adapters_list):
            try:
                logger.info(
                    f"Routing request to provider '{adapter.provider_name}' for capability '{capability}' (Attempt {index + 1})"
                )
                return await adapter.execute(payload)
            except (ProviderTimeoutException, RateLimitException, ProviderUnavailableException) as exc:
                logger.warning(
                    f"Provider '{adapter.provider_name}' failed with {exc.error_code}. Initiating failover..."
                )
                last_exception = exc
                continue
            except Exception as exc:
                logger.error(f"Unrecoverable error on provider '{adapter.provider_name}': {exc}")
                raise

        if last_exception:
            raise last_exception
        raise AIOrchestratorException(
            error_code="AI_PROVIDER_UNAVAILABLE",
            message=f"All provider failover attempts failed for capability: {capability}",
        )
