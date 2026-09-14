import base64
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.db.mongo import MongoConnectionError
from app.models.ai_models import (
    ImageGenerationPromptRequest,
    MediaGenerationRequest,
    PromptRegenerationRequest,
    ScriptGenerationRequest,
    ScriptRegenerationRequest,
    VideoGenerationPromptRequest,
    VideoGenerationRequest,
)
from app.services.cloudinary_service import CloudinaryService
from app.services.ai_workflow_service import AIWorkflowService
from app.services.media_generation_service import MediaGenerationService
from app.utils.http_errors import dependency_error, not_found, provider_error


async def _serialize_uploaded_images(product_images: list[UploadFile] | None):
    if not product_images:
        return None, None

    image_labels: list[str] = []
    image_data_urls: list[str] = []

    for uploaded_file in product_images:
        content_type = uploaded_file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="product_images must be image files.")

        file_bytes = await uploaded_file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="One of the uploaded images is empty.")

        filename = uploaded_file.filename or "uploaded-image"
        image_labels.append(filename)
        image_data_urls.append(
            f"data:{content_type};base64,{base64.b64encode(file_bytes).decode('ascii')}"
        )

    return image_labels, image_data_urls


async def _upload_reference_file(uploaded_file: UploadFile | None, resource_type: str) -> str | None:
    if uploaded_file is None:
        return None

    file_bytes = await uploaded_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded reference file is empty.")

    storage = CloudinaryService()
    if not storage.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Cloudinary is required when uploading reference files. Send a URL instead.",
        )

    suffix = Path(uploaded_file.filename or "").suffix or ".bin"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        result = storage.upload_media(file_source=temp_path, resource_type=resource_type)
        secure_url = result.get("secure_url")
        if not secure_url:
            raise HTTPException(status_code=502, detail="Reference file upload failed.")
        return secure_url
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


from app.services.orchestrator_service import FMFOAIOrchestrator


async def generate_script(
    user_id: str,
    product_name: str,
    product_description: str,
    time_seconds: int,
    product_images: list[UploadFile] | None = None,
):
    orchestrator = FMFOAIOrchestrator()
    try:
        image_labels, image_data_urls = await _serialize_uploaded_images(product_images)
        payload = ScriptGenerationRequest(
            user_id=user_id,
            product_name=product_name,
            product_description=product_description,
            product_images=image_labels,
            time_seconds=time_seconds,
        )
        return await orchestrator.execute_script_workflow(payload, image_data_urls)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def generate_video(payload: VideoGenerationRequest):
    orchestrator = FMFOAIOrchestrator()
    try:
        return await orchestrator.execute_video_workflow(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def regenerate_script(payload: ScriptRegenerationRequest):
    service = AIWorkflowService()
    try:
        return await service.regenerate_script(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def get_job(job_id: str):
    service = AIWorkflowService()
    try:
        job = await service.get_job(job_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    if job is None:
        raise not_found("AI workflow job not found.")
    return job


async def create_image_generation_prompt(payload: ImageGenerationPromptRequest):
    service = MediaGenerationService()
    try:
        return await service.create_image_prompt(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def get_image_generation_prompt(user_id: str, content_id: str):
    service = MediaGenerationService()
    try:
        return await service.get_image_prompt(user_id=user_id, content_id=content_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def regenerate_image_generation_prompt(
    content_id: str, payload: PromptRegenerationRequest
):
    service = MediaGenerationService()
    try:
        return await service.regenerate_image_prompt(user_id=payload.user_id, content_id=content_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def generate_image(payload: MediaGenerationRequest):
    service = MediaGenerationService()
    try:
        return await service.generate_image(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def create_video_generation_prompt(payload: VideoGenerationPromptRequest):
    service = MediaGenerationService()
    try:
        return await service.create_video_prompt(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def get_video_generation_prompt(user_id: str, content_id: str):
    service = MediaGenerationService()
    try:
        return await service.get_video_prompt(user_id=user_id, content_id=content_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def regenerate_video_generation_prompt(
    content_id: str, payload: PromptRegenerationRequest
):
    service = MediaGenerationService()
    try:
        return await service.regenerate_video_prompt(user_id=payload.user_id, content_id=content_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def generate_media_video(payload: MediaGenerationRequest):
    service = MediaGenerationService()
    try:
        return await service.generate_video(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def create_video_edit_prompt(
    *,
    user_id: str,
    prompt: str,
    video_ref: str | None,
    image_ref: str | None,
    audio: bool,
    video_file: UploadFile | None,
    image_file: UploadFile | None,
):
    service = MediaGenerationService()
    try:
        uploaded_video_ref = await _upload_reference_file(video_file, resource_type="video")
        uploaded_image_ref = await _upload_reference_file(image_file, resource_type="image")
        resolved_video_ref = uploaded_video_ref or video_ref
        resolved_image_ref = uploaded_image_ref or image_ref
        if not resolved_video_ref:
            raise HTTPException(status_code=400, detail="video_ref or video_file is required.")
        return await service.create_video_edit_prompt(
            user_id=user_id,
            prompt=prompt,
            video_ref=resolved_video_ref,
            image_ref=resolved_image_ref,
            audio=audio,
        )
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def get_video_edit_prompt(user_id: str, content_id: str):
    service = MediaGenerationService()
    try:
        return await service.get_video_edit_prompt(user_id=user_id, content_id=content_id)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def regenerate_video_edit_prompt(content_id: str, payload: PromptRegenerationRequest):
    service = MediaGenerationService()
    try:
        return await service.regenerate_video_edit_prompt(
            user_id=payload.user_id, content_id=content_id
        )
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def edit_video(payload: MediaGenerationRequest):
    service = MediaGenerationService()
    try:
        return await service.edit_video(payload)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc
