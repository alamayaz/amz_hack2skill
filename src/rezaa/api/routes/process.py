"""Processing endpoints."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from rezaa.api.dependencies import get_pipeline_manager
from rezaa.models.errors import ValidationError
from rezaa.models.preferences import UserPreferences
from rezaa.pipeline.manager import PipelineManager

router = APIRouter(prefix="/api/v1", tags=["process"])


class ProcessRequest(BaseModel):
    job_id: str
    preferences: UserPreferences = Field(default_factory=UserPreferences)


@router.post("/process")
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Start video processing for a job."""
    state = manager.get_job_state(request.job_id)
    if not state:
        raise ValidationError(f"Job {request.job_id} not found")

    if not state.audio_path:
        raise ValidationError("No audio file uploaded for this job")
    if not state.video_paths:
        raise ValidationError("No video files uploaded for this job")

    background_tasks.add_task(
        manager.process,
        job_id=request.job_id,
        audio_path=Path(state.audio_path),
        video_paths=[Path(p) for p in state.video_paths],
        preferences=request.preferences,
    )

    return {
        "job_id": request.job_id,
        "status": "processing",
        "message": "Processing started",
    }


@router.delete("/process/{job_id}")
async def cancel_processing(
    job_id: str,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Cancel a running processing job."""
    cancelled = manager.cancel_job(job_id)
    if not cancelled:
        raise ValidationError(f"Job {job_id} not found")
    return {"job_id": job_id, "status": "cancelled"}
