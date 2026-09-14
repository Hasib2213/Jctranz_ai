from __future__ import annotations

from typing import Any

from app.core.config import get_settings


class CostService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def format_usd(self, amount: float | int | None) -> str:
        value = float(amount or 0.0)
        if value <= 0:
            return "$0.00"
        if value >= 1:
            return f"${value:.2f}"
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        if "." not in text:
            text = f"{text}.00"
        return f"${text}"

    def calculate_openai_cost(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> float:
        input_cost = (
            input_tokens / 1_000_000 * self.settings.openai_input_price_per_1m_tokens
        )
        output_cost = (
            output_tokens / 1_000_000 * self.settings.openai_output_price_per_1m_tokens
        )
        return round(input_cost + output_cost, 8)

    def extract_openai_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(
                usage.get("output_tokens") or usage.get("completion_tokens") or 0
            )
        else:
            input_tokens = int(
                getattr(usage, "input_tokens", None)
                or getattr(usage, "prompt_tokens", None)
                or 0
            )
            output_tokens = int(
                getattr(usage, "output_tokens", None)
                or getattr(usage, "completion_tokens", None)
                or 0
            )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def openai_cost_detail(self, response: Any) -> dict[str, Any]:
        usage = self.extract_openai_usage(response)
        cost = self.calculate_openai_cost(
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]
        )
        return {
            **usage,
            "cost": cost,
            "formatted_cost": self.format_usd(cost),
        }

    def fal_image_cost(self, image_count: int = 1, provider_response: Any = None) -> float:
        actual = self.extract_provider_cost(provider_response)
        if actual is not None:
            return actual
        return round(max(image_count, 1) * self.settings.fal_image_price_usd, 8)

    def fal_video_cost(
        self,
        duration_seconds: int,
        provider_response: Any = None,
        audio: bool = True,
    ) -> float:
        actual = self.extract_provider_cost(provider_response)
        if actual is not None:
            return actual

        price = self.settings.fal_video_price_per_second_usd
        if audio and self.settings.fal_video_audio_on_price_per_second_usd:
            price = self.settings.fal_video_audio_on_price_per_second_usd
        elif not audio and self.settings.fal_video_audio_off_price_per_second_usd:
            price = self.settings.fal_video_audio_off_price_per_second_usd

        return round(duration_seconds * price, 8)

    def fal_video_edit_cost(
        self, duration_seconds: int | None = None, provider_response: Any = None
    ) -> float:
        actual = self.extract_provider_cost(provider_response)
        if actual is not None:
            return actual
        seconds = duration_seconds or self.settings.fal_video_duration_seconds
        return round(seconds * self.settings.fal_video_edit_price_per_second_usd, 8)

    def extract_provider_cost(self, value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, dict):
            for key in ("cost", "provider_cost", "total_cost", "amount"):
                raw = value.get(key)
                parsed = self._parse_number(raw)
                if parsed is not None:
                    return parsed

            for nested_key in ("usage", "billing", "pricing", "metrics"):
                nested_cost = self.extract_provider_cost(value.get(nested_key))
                if nested_cost is not None:
                    return nested_cost

            return None

        if isinstance(value, list):
            total = 0.0
            found = False
            for item in value:
                item_cost = self.extract_provider_cost(item)
                if item_cost is not None:
                    total += item_cost
                    found = True
            return round(total, 8) if found else None

        return self._parse_number(value)

    def extract_magica_credits(
        self, provider_response: Any, fallback_cost_usd: float = 0.0
    ) -> float:
        if isinstance(provider_response, dict):
            for key in ("creditUsed", "credit_used", "credits_used", "credits"):
                raw = provider_response.get(key)
                if raw is not None:
                    parsed = self._parse_number(raw)
                    if parsed is not None:
                        return round(parsed, 4)

            clips = (
                provider_response.get("clip_responses")
                or provider_response.get("chunk_results")
                or []
            )
            total = 0.0
            found = False
            for clip in clips:
                if isinstance(clip, dict):
                    for key in ("creditUsed", "credit_used", "credits_used", "credits"):
                        raw = clip.get(key)
                        if raw is not None:
                            parsed = self._parse_number(raw)
                            if parsed is not None:
                                total += parsed
                                found = True
                                break
            if found:
                return round(total, 4)

            run = provider_response.get("run")
            if isinstance(run, dict):
                raw = run.get("creditUsed")
                parsed = self._parse_number(raw)
                if parsed is not None:
                    return round(parsed, 4)

        if fallback_cost_usd > 0:
            return float(round(fallback_cost_usd * 1_000_000, 4))

        return 0.0


    def _parse_number(self, raw: Any) -> float | None:
        if raw is None or isinstance(raw, bool):
            return None
        if isinstance(raw, int | float):
            return float(raw)
        if isinstance(raw, str):
            cleaned = raw.replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

