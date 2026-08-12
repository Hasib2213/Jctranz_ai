from fastapi import APIRouter

from app.controllers import ai_controller
from app.models.ai_models import (
    ScriptGenerationRequest,
    ScriptGenerationResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)

router = APIRouter(prefix="/ai", tags=["AI Workflow"])


@router.post("/scripts", response_model=ScriptGenerationResponse)
async def create_promotional_script(payload: ScriptGenerationRequest):
    return await ai_controller.generate_script(payload)


@router.post("/videos", response_model=VideoGenerationResponse)
async def create_promotional_video(payload: VideoGenerationRequest):
    return await ai_controller.generate_video(payload)


@router.get("/jobs/{job_id}")
async def read_workflow_job(job_id: str):
    return await ai_controller.get_job(job_id)
