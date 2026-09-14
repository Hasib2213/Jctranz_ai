from enum import Enum
from fastapi import HTTPException, status


class AIErrorCode(str, Enum):
    AI_PROVIDER_TIMEOUT = "AI_PROVIDER_TIMEOUT"
    AI_PROVIDER_UNAVAILABLE = "AI_PROVIDER_UNAVAILABLE"
    AI_RATE_LIMIT = "AI_RATE_LIMIT"
    AI_GENERATION_FAILED = "AI_GENERATION_FAILED"
    AI_INVALID_REQUEST = "AI_INVALID_REQUEST"
    AI_INSUFFICIENT_CREDITS = "AI_INSUFFICIENT_CREDITS"


class AIOrchestratorException(HTTPException):
    def __init__(
        self,
        error_code: AIErrorCode,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict | None = None,
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        super().__init__(
            status_code=status_code,
            detail={
                "error_code": self.error_code.value,
                "message": self.message,
                "details": self.details,
            },
        )


class InsufficientCreditsException(AIOrchestratorException):
    def __init__(self, message: str = "Insufficient user credits to perform AI operation."):
        super().__init__(
            error_code=AIErrorCode.AI_INSUFFICIENT_CREDITS,
            message=message,
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
        )


class RateLimitException(AIOrchestratorException):
    def __init__(self, message: str = "AI provider rate limit reached."):
        super().__init__(
            error_code=AIErrorCode.AI_RATE_LIMIT,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


class ProviderTimeoutException(AIOrchestratorException):
    def __init__(self, message: str = "AI provider request timed out."):
        super().__init__(
            error_code=AIErrorCode.AI_PROVIDER_TIMEOUT,
            message=message,
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )


class ProviderUnavailableException(AIOrchestratorException):
    def __init__(self, message: str = "AI provider service is currently unavailable."):
        super().__init__(
            error_code=AIErrorCode.AI_PROVIDER_UNAVAILABLE,
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class GenerationFailedException(AIOrchestratorException):
    def __init__(self, message: str = "AI generation process failed."):
        super().__init__(
            error_code=AIErrorCode.AI_GENERATION_FAILED,
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
