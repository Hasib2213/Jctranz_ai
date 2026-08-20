import base64

from fastapi import HTTPException, UploadFile

from app.db.mongo import MongoConnectionError
from app.models.ai_models import ScriptGenerationRequest, VideoGenerationRequest
from app.services.ai_workflow_service import AIWorkflowService
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


async def generate_script(
    user_id: str,
    product_name: str,
    product_description: str,
    time_seconds: int,
    product_images: list[UploadFile] | None = None,
):
    service = AIWorkflowService()
    try:
        image_labels, image_data_urls = await _serialize_uploaded_images(product_images)
        payload = ScriptGenerationRequest(
            user_id=user_id,
            product_name=product_name,
            product_description=product_description,
            product_images=image_labels,
            time_seconds=time_seconds,
        )
        return await service.generate_script(payload, image_data_urls)
    except MongoConnectionError as exc:
        raise dependency_error(str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise provider_error(str(exc)) from exc


async def generate_video(payload: VideoGenerationRequest):
    service = AIWorkflowService()
    try:
        return await service.generate_video(payload)
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
