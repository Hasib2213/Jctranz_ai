from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    SCRIPT_GENERATED = "script_generated"
    VIDEO_GENERATING = "video_generating"
    VIDEO_COMPLETED = "video_completed"
    FAILED = "failed"


class ScriptScene(BaseModel):
    sequence: int = Field(..., ge=1)
    time: str = Field(..., min_length=1, max_length=40)
    visual: str = Field(..., min_length=1, max_length=1000)
    voiceover: str = Field(..., min_length=1, max_length=1000)


class ScriptGenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    product_name: str = Field(..., min_length=2, max_length=120)
    product_description: str = Field(..., min_length=10, max_length=2000)
    product_images: list[str] | None = None
    time_seconds: int = Field(default=15, ge=5, le=30)


class ScriptGenerationResponse(BaseModel):
    job_id: str
    promotional_script: list[ScriptScene]
    status: JobStatus = JobStatus.SCRIPT_GENERATED


class ScriptRegenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    job_id: str = Field(..., min_length=1, max_length=120)
    promotional_script: list[ScriptScene] = Field(..., min_length=1)


class VideoGenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    job_id: str = Field(..., min_length=1, max_length=120)


class VideoGenerationContext(BaseModel):
    product_name: str = Field(..., min_length=2, max_length=120)
    product_image_url: str | None = None
    approved_script: str = Field(..., min_length=10, max_length=3000)
    video_style: str = Field(default="UGC product promo", max_length=200)
    voice_style: str = Field(default="natural friendly voiceover", max_length=200)
    music_style: str = Field(default="clean upbeat background music", max_length=200)
    time_seconds: int = Field(default=15, ge=5, le=30)


class VideoGenerationResponse(BaseModel):
    job_id: str
    video_url: str
    cdn_video_url: str | None = None
    cloudinary_asset: dict[str, Any] | None = None
    provider_response: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.VIDEO_COMPLETED


class WorkflowJob(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    status: JobStatus
    user_id: str
    product_name: str
    product_description: str | None = None
    product_images: list[str] | None = None
    product_image_data_urls: list[str] | None = None
    product_image_url: str | None = None
    time_seconds: int | None = None
    script_history: list[list[ScriptScene]] | None = None
    promotional_script: list[ScriptScene] | None = None
    approved_script_text: str | None = None
    final_video_prompt: str | None = None
    video_generation_mode: str | None = None
    video_url: str | None = None
    cdn_video_url: str | None = None
    cloudinary_asset: dict[str, Any] | None = None
    provider_response: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True


class AIGenerationType(str, Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    VIDEO_EDIT = "video_edit"


class CostSummary(BaseModel):
    openAI_cost: str = "$0.00"
    falAI_cost: str = "$0.00"
    total_cost: str = "$0.00"


class PromptVersion(BaseModel):
    version: int
    ai_refined_prompt: str = Field(alias="AI_refine_prompt")
    openAI_cost: str = "$0.00"
    created_at: datetime | None = None

    model_config = ConfigDict(populate_by_name=True)


class ImageGenerationPromptRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=3000)
    resolution: str = Field(..., min_length=1, max_length=80)
    aspect_ratio: str = Field(..., min_length=1, max_length=40)


class ImageGenerationPromptResponse(BaseModel):
    user_id: str
    content_id: str
    ai_refined_prompt: str = Field(alias="AI_refine_prompt")
    user_prompt: str
    resolution: str
    aspect_ratio: str
    openAI_cost: str = "$0.00"

    model_config = ConfigDict(populate_by_name=True)


class ImageGenerationPromptDetailResponse(ImageGenerationPromptResponse):
    prompt_versions: list[PromptVersion] = Field(default_factory=list)
    openAI_cost_total: str = "$0.00"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class VideoGenerationPromptRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=3000)
    resolution: str = Field(..., min_length=1, max_length=80)
    aspect_ratio: str = Field(..., min_length=1, max_length=40)
    time: Literal[5, 10, 15, 20, 30]
    audio: bool = True


class VideoGenerationPromptResponse(BaseModel):
    user_id: str
    content_id: str
    ai_refined_prompt: str = Field(alias="AI_refine_prompt")
    user_prompt: str
    resolution: str
    aspect_ratio: str
    time: int
    audio: bool
    openAI_cost: str = "$0.00"

    model_config = ConfigDict(populate_by_name=True)


class VideoGenerationPromptDetailResponse(VideoGenerationPromptResponse):
    prompt_versions: list[PromptVersion] = Field(default_factory=list)
    openAI_cost_total: str = "$0.00"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PromptRegenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)


class VideoEditPromptResponse(BaseModel):
    user_id: str
    content_id: str
    ai_refined_prompt: str = Field(alias="AI_refine_prompt")
    user_prompt: str
    video_ref: str
    image_ref: str | None = None
    audio: bool
    openAI_cost: str = "$0.00"

    model_config = ConfigDict(populate_by_name=True)


class VideoEditPromptDetailResponse(VideoEditPromptResponse):
    prompt_versions: list[PromptVersion] = Field(default_factory=list)
    openAI_cost_total: str = "$0.00"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MediaGenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)
    content_id: str = Field(..., min_length=1, max_length=120)


class MediaGenerationResponse(BaseModel):
    user_id: str
    generated_content_id: str
    content_url: str
    openAI_cost: str
    falAI_cost: str
    total_cost: str
