from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Jctranz AI Workflow"
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"

    openai_api_key: str = ""
    openai_script_model: str = "gpt-4o-mini"
    openai_prompt_model: str = "gpt-4o-mini"
    openai_input_price_per_1m_tokens: float = 0.0
    openai_output_price_per_1m_tokens: float = 0.0

    fal_key: str = ""
    fal_image_model: str = "fal-ai/flux-pro/v1.1"
    fal_video_model: str = "fal-ai/kling-video/v2.6/pro/image-to-video"
    fal_text_video_model: str = "fal-ai/kling-video/v2.6/pro/text-to-video"
    fal_video_edit_model: str = "fal-ai/kling-video/o1/video-to-video/edit"
    fal_video_duration_seconds: int = 15
    fal_image_price_usd: float = 0.0
    fal_video_price_per_second_usd: float = 0.0
    fal_video_audio_off_price_per_second_usd: float = 0.0
    fal_video_audio_on_price_per_second_usd: float = 0.0
    fal_video_edit_price_per_second_usd: float = 0.0

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "jctranz_ai"

    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    cloudinary_folder: str = "jctranz_ai_assets"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
