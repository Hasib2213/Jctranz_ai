from fastapi import HTTPException

from app.db.mongo import MongoConnectionError
from app.models.ai_models import ScriptGenerationRequest, VideoGenerationRequest
from app.services.ai_workflow_service import AIWorkflowService
from app.utils.http_errors import dependency_error, not_found, provider_error


async def generate_script(payload: ScriptGenerationRequest):
    service = AIWorkflowService()
    try:
        return await service.generate_script(payload)
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
