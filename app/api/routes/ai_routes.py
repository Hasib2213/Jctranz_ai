from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.controllers import ai_controller
from app.models.ai_models import ScriptGenerationResponse, VideoGenerationRequest, VideoGenerationResponse

router = APIRouter(prefix="/ai", tags=["AI Workflow"])


@router.post("/scripts", response_model=ScriptGenerationResponse)
async def create_promotional_script(
    user_id: Annotated[str, Form(...)],
    product_name: Annotated[str, Form(...)],
    product_description: Annotated[str, Form(...)],
    time_seconds: Annotated[int, Form()] = 15,
    product_images: Annotated[list[UploadFile] | None, File()] = None,
):
    return await ai_controller.generate_script(
        user_id=user_id,
        product_name=product_name,
        product_description=product_description,
        time_seconds=time_seconds,
        product_images=product_images,
    )


@router.post("/videos", response_model=VideoGenerationResponse)
async def create_promotional_video(payload: VideoGenerationRequest):
    return await ai_controller.generate_video(payload)


@router.get("/jobs/{job_id}")
async def read_workflow_job(job_id: str):
    return await ai_controller.get_job(job_id)
