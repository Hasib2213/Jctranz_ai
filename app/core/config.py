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

    magica_api_key: str = ""
    magica_base_url: str = "https://inference.magica.com/v1"
    magica_image_provider_enabled: bool = True
    magica_image_model_node_type: str = "flux_2_max"
    magica_text_image_submodel: str = "flux-2-max-text"
    magica_image_output_format: str = "JPEG"
    magica_video_provider_enabled: bool = True
    magica_video_model_node_type: str = "seedance_2_0_fast"
    magica_text_video_submodel: str = "seedance-2.0-fast-text-to-video"
    magica_image_video_submodel: str = "seedance-2.0-fast-image-to-video"
    magica_video_edit_provider_enabled: bool = True
    magica_video_edit_model_node_type: str = "kling_o3_pro_video_edit"
    magica_poll_interval_seconds: float = 3.0
    magica_timeout_seconds: int = 900
    magica_480p_price_per_second_usd: float = 0.1076
    magica_720p_price_per_second_usd: float = 0.2419

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
