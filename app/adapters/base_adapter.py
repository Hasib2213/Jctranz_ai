from abc import ABC, abstractmethod
from typing import Any


class BaseAIAdapter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns provider identification string (e.g. 'openai', 'fal_ai')."""
        pass

    @property
    @abstractmethod
    def capability(self) -> str:
        """Returns capability string ('script', 'video', 'image', 'voice', 'music', 'chatbot')."""
        pass

    @abstractmethod
    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Executes the AI provider request and returns normalized response dict.
        Must normalize exceptions to app.core.errors classes.
        """
        pass
