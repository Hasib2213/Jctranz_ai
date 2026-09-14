from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.controllers import ai_controller
from app.models.ai_models import (
    ImageGenerationPromptDetailResponse,
    ImageGenerationPromptRequest,
    ImageGenerationPromptResponse,
    MediaGenerationRequest,
    MediaGenerationResponse,
    PromptRegenerationRequest,
    VideoEditPromptDetailResponse,
    VideoEditPromptResponse,
    VideoGenerationPromptDetailResponse,
    VideoGenerationPromptRequest,
    VideoGenerationPromptResponse,
)

router = APIRouter(prefix="/ai", tags=["AI Workflow"])


@router.post("/image-generation-prompts", response_model=ImageGenerationPromptResponse)
async def create_image_generation_prompt(payload: ImageGenerationPromptRequest):
    return await ai_controller.create_image_generation_prompt(payload)


@router.get(
    "/image-generation-prompts/{content_id}",
    response_model=ImageGenerationPromptDetailResponse,
)
async def read_image_generation_prompt(content_id: str, user_id: str):
    return await ai_controller.get_image_generation_prompt(
        user_id=user_id,
        content_id=content_id,
    )


@router.post(
    "/image-generation-prompts/{content_id}/regenerate",
    response_model=ImageGenerationPromptDetailResponse,
)
async def regenerate_image_generation_prompt(
    content_id: str, payload: PromptRegenerationRequest
):
    return await ai_controller.regenerate_image_generation_prompt(content_id, payload)


@router.post("/image-generations", response_model=MediaGenerationResponse)
async def create_image_generation(payload: MediaGenerationRequest):
    return await ai_controller.generate_image(payload)


@router.post("/video-generation-prompts", response_model=VideoGenerationPromptResponse)
async def create_video_generation_prompt(payload: VideoGenerationPromptRequest):
    return await ai_controller.create_video_generation_prompt(payload)


@router.get(
    "/video-generation-prompts/{content_id}",
    response_model=VideoGenerationPromptDetailResponse,
)
async def read_video_generation_prompt(content_id: str, user_id: str):
    return await ai_controller.get_video_generation_prompt(
        user_id=user_id,
        content_id=content_id,
    )


@router.post(
    "/video-generation-prompts/{content_id}/regenerate",
    response_model=VideoGenerationPromptDetailResponse,
)
async def regenerate_video_generation_prompt(
    content_id: str, payload: PromptRegenerationRequest
):
    return await ai_controller.regenerate_video_generation_prompt(content_id, payload)


@router.post("/video-generations", response_model=MediaGenerationResponse)
async def create_media_video_generation(payload: MediaGenerationRequest):
    return await ai_controller.generate_media_video(payload)


@router.post("/video-edit-prompts", response_model=VideoEditPromptResponse)
async def create_video_edit_prompt(
    user_id: Annotated[str, Form(...)],
    prompt: Annotated[str, Form(...)],
    audio: Annotated[bool, Form()] = True,
    video_ref: Annotated[str | None, Form()] = None,
    image_ref: Annotated[str | None, Form()] = None,
    video_file: Annotated[UploadFile | None, File()] = None,
    image_file: Annotated[UploadFile | None, File()] = None,
):
    return await ai_controller.create_video_edit_prompt(
        user_id=user_id,
        prompt=prompt,
        video_ref=video_ref,
        image_ref=image_ref,
        audio=audio,
        video_file=video_file,
        image_file=image_file,
    )


@router.get("/video-edit-prompts/{content_id}", response_model=VideoEditPromptDetailResponse)
async def read_video_edit_prompt(content_id: str, user_id: str):
    return await ai_controller.get_video_edit_prompt(
        user_id=user_id,
        content_id=content_id,
    )


@router.post(
    "/video-edit-prompts/{content_id}/regenerate",
    response_model=VideoEditPromptDetailResponse,
)
async def regenerate_video_edit_prompt(content_id: str, payload: PromptRegenerationRequest):
    return await ai_controller.regenerate_video_edit_prompt(content_id, payload)


@router.post("/video-edits", response_model=MediaGenerationResponse)
async def create_video_edit(payload: MediaGenerationRequest):
    return await ai_controller.edit_video(payload)
