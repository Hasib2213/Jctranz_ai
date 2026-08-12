from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    PENDING = "pending"
    SCRIPT_GENERATED = "script_generated"
    VIDEO_GENERATING = "video_generating"
    VIDEO_COMPLETED = "video_completed"
    FAILED = "failed"


class ScriptGenerationRequest(BaseModel):
    product_name: str = Field(..., min_length=2, max_length=120)
    product_description: str = Field(..., min_length=10, max_length=2000)
    creator_prompt: str | None = Field(default=None, max_length=1500)
    target_audience: str | None = Field(default=None, max_length=300)
    tone: str | None = Field(default="friendly, persuasive, UGC-style", max_length=200)
    duration_seconds: int = Field(default=15, ge=5, le=30)


class ScriptGenerationResponse(BaseModel):
    job_id: str
    promotional_script: str
    status: JobStatus = JobStatus.SCRIPT_GENERATED


class VideoGenerationRequest(BaseModel):
    job_id: str | None = None
    product_name: str = Field(..., min_length=2, max_length=120)
    product_image_url: HttpUrl
    approved_script: str = Field(..., min_length=10, max_length=3000)
    video_style: str = Field(default="UGC product promo", max_length=200)
    voice_style: str = Field(default="natural friendly voiceover", max_length=200)
    music_style: str = Field(default="clean upbeat background music", max_length=200)
    duration_seconds: int = Field(default=15, ge=5, le=30)


class VideoGenerationResponse(BaseModel):
    job_id: str
    video_url: str
    provider_response: dict[str, Any] = Field(default_factory=dict)
    status: JobStatus = JobStatus.VIDEO_COMPLETED


class WorkflowJob(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    status: JobStatus
    product_name: str
    product_description: str | None = None
    product_image_url: str | None = None
    creator_prompt: str | None = None
    promotional_script: str | None = None
    final_video_prompt: str | None = None
    video_url: str | None = None
    provider_response: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
